# 
# minimalistic implementation of Flash Attention 3 for learning purposes.
# 
# note: there is zero reason that you would not be able to execute on this within 1 day.
# you've basically learned all the fundamentals. if you need to uncover something more,
# you know the playbook. just isolate the component, make minimal tests, play around with it.
# imagine how you used to learn python when you were 10 years old. execution feedback, isolated environments
# understand how things work deeply, and what is going on.
# although benchmarking might take longer, you should be able to do the bulk of the kernel work in 1 day.
# this is not a problem, and there is basically a playbook for every single section.

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warpgroup import OperandSource
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import torch
import math
import cuda.bindings.driver as cuda
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from typing import List, Type, Tuple

def transpose(tensor: cutlass.Tensor, new_indices: List[int]):
    shape = tuple(tensor.shape[i] for i in new_indices)
    stride = tuple(tensor.layout.stride[i] for i in new_indices)
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


# H100 kernel for Flash Attention 3
class HopperFA2:
    def __init__(
        self,
        cta_tile: Tuple[int, int, int],
    ):
        # tiler for the first part.
        self.cta_tile = cta_tile
        self.kv_pipeline_stages = 4
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        q: cutlass.Tensor, # (B, H_qo, S_qo, D) [DANGER: transpose later on]
        k: cutlass.Tensor, # (B, H_kv, S_kv, D) [DANGER: transpose later on]
        v: cutlass.Tensor, # (B, H_kv, S_kv, D) [DANGER: transpose later on]
        o: cutlass.Tensor, # (B, H_qo, S_qo, D) [DANGER: transpose later on]
        stream: cuda.CUstream,
    ):
        q = transpose(q, [2, 3, 1, 0]) # (S_qo, D, H_qo, B)
        k = transpose(k, [2, 3, 1, 0]) # (S_kv, D, H_kv, B)
        v = transpose(v, [2, 3, 1, 0]) # (S_kv, D, H_kv, B)
        o = transpose(o, [2, 3, 1, 0]) # (S_qo, D, H_qo, B)

        q_smem_shape = (self.cta_tile[0], self.cta_tile[2])
        k_smem_shape = (self.cta_tile[1], self.cta_tile[2])
        v_smem_shape = (self.cta_tile[1], self.cta_tile[2])

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type

        def make_smem_layout(smem_shape: Tuple[int, int], dtype: Type[cutlass.Numeric], stages: int):
            smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
                dtype,
            )
            return cute.tile_to_shape(
                smem_layout_atom,
                cute.append(smem_shape, stages),
                order=(0, 1, 2)
            )

        # 1. initialize shared-memory layout.
        self.q_smem_layout_staged = make_smem_layout(q_smem_shape, self.q_dtype, stages=1)
        self.k_smem_layout_staged = make_smem_layout(k_smem_shape, self.k_dtype, self.kv_pipeline_stages)
        self.v_smem_layout_staged = make_smem_layout(v_smem_shape, self.v_dtype, self.kv_pipeline_stages)
        self.q_smem_layout_unstaged = cute.slice_(self.q_smem_layout_staged, (None, None, 0))
        self.k_smem_layout_unstaged = cute.slice_(self.k_smem_layout_staged, (None, None, 0))
        self.v_smem_layout_unstaged = cute.slice_(self.v_smem_layout_staged, (None, None, 0))

        # 2. setup tma atoms and tensors.
        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        q_tma_atom, q_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            q,
            self.q_smem_layout_unstaged,
            (self.cta_tile[0], self.cta_tile[2])
        )
        k_tma_atom, k_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            k,
            self.k_smem_layout_unstaged,
            (self.cta_tile[1], self.cta_tile[2])
        )
        v_tma_atom, v_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            v,
            self.v_smem_layout_unstaged,
            (self.cta_tile[1], self.cta_tile[2])
        )

        # 4. setup shared storage.

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            mi_ptr: cute.struct.MemRange[cutlass.Int32, self.cta_tile[0]]
            li_ptr: cute.struct.MemRange[cutlass.Int32, self.cta_tile[0]]
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_pipeline_stages * 2]

        self.shared_storage = SharedStorage

        self.load_warp_id = 0
        self.mma_warp_ids = [4, 5, 6, 7]

        mm = cute.nvgpu.warpgroup.OperandMajorMode.K
        self.atom_layout_mnk = (1, 1, 1)
        self.acc_dtype = cutlass.Float32
        self.tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            mm,
            mm,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(self.cta_tile[0], self.cta_tile[1]),
        )

        self.tiled_mma_v = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.v_dtype,
            mm,
            mm,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(self.cta_tile[1], self.cta_tile[2]),
            a_source=OperandSource.RMEM,
        )

        self.kernel(
            q_tma_tensor,
            k_tma_tensor,
            v_tma_tensor,
            self.q_smem_layout_staged,
            self.k_smem_layout_staged,
            self.v_smem_layout_staged,
            q_tma_atom,
            k_tma_atom,
            v_tma_atom,
            self.tiled_mma_qk,
            self.tiled_mma_v,
        ).launch(
            grid=[1, 1, 1],
            block=[256, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gQ: cutlass.Tensor,
        gK: cutlass.Tensor,
        gV: cutlass.Tensor,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        q_tma_atom: cute.Atom,
        k_tma_atom: cute.Atom,
        v_tma_atom: cute.Atom,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_v: cute.TiledMma,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(q_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(k_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(v_tma_atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ: cute.Tensor = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK: cute.Tensor = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV: cute.Tensor = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )

        UNUSED = 1
        load_q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, UNUSED)
        load_q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=load_q_producer_group,
            consumer_group=load_q_consumer_group,
            tx_count=cute.size_in_bytes(self.q_dtype, cute.slice_(q_smem_layout_staged, (None, None, 0))),
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
        )

        load_kv_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len([self.load_warp_id]))
        load_kv_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.mma_warp_ids))
        load_kv_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_pipeline_stages,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=cute.size_in_bytes(self.k_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0))) + cute.size_in_bytes(self.v_dtype, cute.slice_(v_smem_layout_staged, (None, None, 0))),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        )

        gQ_tiled = cute.flat_divide(gQ, (self.cta_tile[0], self.cta_tile[2])) # (S_qo, D, H_qo, B)
        gK_tiled = cute.flat_divide(gK, (self.cta_tile[1], self.cta_tile[2])) # (S_kv, D, H_kv, B)
        gV_tiled = cute.flat_divide(gV, (self.cta_tile[1], self.cta_tile[2])) # (S_kv, D, H_kv, B)
        gQ_local = gQ_tiled[*(None, None), *(bidx, None, bidy, bidz)] # (tile_M, tile_N, 1)
        gK_local = gK_tiled[*(None, None), *(None, 0, bidy, bidz)] # (tile_M, tile_N, S_kv_blocks)
        gV_local = gV_tiled[*(None, None), *(None, 0, bidy, bidz)] # (tile_M, tile_N, S_kv_blocks)

        # load q into memory.
        is_producer_warp = warp_idx in [self.load_warp_id]
        is_consumer_warp = warp_idx in self.mma_warp_ids

        # partition tensors for TMA
        cluster_shape_mnk = (1, 1, 1)
        cta_layout = cute.make_layout(cluster_shape_mnk)
        cta_layout = cute.make_layout(cute.slice_(cta_layout, (0, None, 0)).shape)
        cta_crd = (0,)

        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ_local, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            k_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK_local, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            v_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV_local, 0, 2),
        )

        num_kv_blocks = cute.size(gK_local, mode=[2])

        if is_producer_warp:
            # load q 
            load_q_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            load_q_pipeline.producer_acquire(load_q_pipeline_state)
            cute.copy(
                q_tma_atom,
                tQgQ[(None, 0)],
                tQsQ[(None, 0)],
                tma_bar_ptr=load_q_pipeline.producer_get_barrier(load_q_pipeline_state)
            )

            load_kv_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_pipeline_stages)
            for i in cutlass.range(num_kv_blocks, unroll=1):
                load_kv_pipeline.producer_acquire(load_kv_pipeline_state)

                sK_chunk = tKsK[(None, load_kv_pipeline_state.index)]
                sV_chunk = tVsV[(None, load_kv_pipeline_state.index)]
                gK_chunk = tKgK[(None, load_kv_pipeline_state.count)]
                gV_chunk = tVgV[(None, load_kv_pipeline_state.count)]

                cute.copy(
                    k_tma_atom,
                    gK_chunk,
                    sK_chunk,
                    tma_bar_ptr=load_kv_pipeline.producer_get_barrier(load_kv_pipeline_state)
                )
                cute.copy(
                    v_tma_atom,
                    gV_chunk,
                    sV_chunk,
                    tma_bar_ptr=load_kv_pipeline.producer_get_barrier(load_kv_pipeline_state)
                )
                load_kv_pipeline_state.advance()


        qk_thr_mma = tiled_mma_qk.get_slice(tidx)
        tCsK = qk_thr_mma.partition_A(sK)
        tCsQ = qk_thr_mma.partition_B(sQ)
        tCrK = tiled_mma_qk.make_fragment_A(tCsK)
        tCrQ = tiled_mma_qk.make_fragment_B(tCsQ)

        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.cta_tile[0], self.cta_tile[1])
        )
        tStS = cute.make_fragment(qk_acc_shape, self.acc_dtype)

        # note that we only properly partition B and C.
        # for A, we will just re-format the output later on.
        v_thr_mma = tiled_mma_v.get_slice(tidx)
        tCsV = v_thr_mma.partition_B(sV)

        tCrV = tiled_mma_v.make_fragment_B(tCsV)

        o_acc_shape = v_thr_mma.partition_shape_C(
            (self.cta_tile[1], self.cta_tile[2])
        )
        v_A_shape = v_thr_mma.partition_shape_A(
            (self.cta_tile[0], self.cta_tile[1])
        )
        tStO = cute.make_fragment(o_acc_shape, self.acc_dtype)
        tOtO = cute.make_fragment(o_acc_shape, self.acc_dtype)

        #
        if is_consumer_warp:
            # run mma
            mma_kv_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_pipeline_stages)
            mma_q_pipeline_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            load_q_pipeline.consumer_wait(mma_q_pipeline_state)
            load_q_pipeline.consumer_release(mma_q_pipeline_state)

            # each thread contains the accumulator outputs from 2 separate accumulators.
            # we need two register accumulators to store the running max and sum for each of the rows.

            cumm_row_max_1 = -cutlass.Float32.inf
            cumm_row_max_2 = -cutlass.Float32.inf

            for k_block in cutlass.range(num_kv_blocks, unroll=1):
                load_kv_pipeline.consumer_wait(mma_kv_pipeline_state)

                num_k_microtiles = cute.size(tCrK, mode=[2])

                tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k_microtile_ix in cutlass.range(num_k_microtiles, unroll=1):
                    q_block_coord = (None, None, k_microtile_ix, 0)
                    k_block_coord = (None, None, k_microtile_ix, mma_kv_pipeline_state.index)
                    cute.gemm(
                        tiled_mma_qk,
                        tStS,
                        tCrK[k_block_coord],
                        tCrQ[k_block_coord],
                        tStS,
                    )
                    if k_microtile_ix == 0:
                        tiled_mma_qk.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                # use a composition to get the desired matrices that you want.
                FP32_bytes = 4
                tStS_indexer = cute.make_layout((2, cute.size(tStS) // 4), stride=(1, 4))
                row_layouts = cute.composition(tStS.layout, tStS_indexer)

                # each thread contains 2 rows from the MMA outputs.
                row_1 = cute.make_tensor(tStS.iterator, row_layouts)
                row_2 = cute.make_tensor(tStS.iterator + 2 * FP32_bytes, row_layouts)

                row_1_max_ssa = row_1.load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                row_2_max_ssa = row_2.load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)

                for i in cutlass.range_constexpr(2):
                    row_1_max_ssa = cutlass.max(row_1_max_ssa, cute.arch.shuffle_sync_bfly(row_1_max_ssa, 1 << i))
                    row_2_max_ssa = cutlass.max(row_2_max_ssa, cute.arch.shuffle_sync_bfly(row_2_max_ssa, 1 << i))
                cumm_row_max_1 = cutlass.max(cumm_row_max_1, row_1_max_ssa)
                cumm_row_max_2 = cutlass.max(cumm_row_max_2, row_2_max_ssa)

                log2_e = math.log2(math.e)
                row_1_exp = cute.exp2((row_1.load() - row_1_max_ssa) * log2_e)
                row_2_exp = cute.exp2((row_2.load() - row_1_max_ssa) * log2_e)

                # store this back into row_1 and row_2 of tStS.
                # and rename tStS to tPtP for the future.
                row_1.store(row_1_exp)
                row_2.store(row_2_exp)
                tPtP = tStS

                # we re-format the tPtP we have into the shape that we need.
                p_A_layout = cute.make_layout(v_A_shape)
                p_A_tensor_fp32 = cute.make_tensor(tPtP.iterator, p_A_layout)
                p_A_tensor = cute.make_fragment(p_A_tensor_fp32.layout, self.k_dtype)

                # convert to fp32
                p_A_tensor_fp32.store(p_A_tensor.load().to(self.k_dtype))

                # perform the GEMM.
                tiled_mma_v.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                num_k_microtiles = cute.size(p_A_tensor, mode=[2])
                for k_microtile_ix in cutlass.range(num_k_microtiles, unroll=1):
                    p_block_coord = (None, None, k_microtile_ix)
                    v_block_coord = (None, None, k_microtile_ix, mma_kv_pipeline_state.index)
                    cute.gemm(
                        tiled_mma_v,
                        tStO,
                        p_A_tensor[p_block_coord],
                        tCrV[v_block_coord],
                        tStO,
                    )
                    if k_microtile_ix == 0:
                        tiled_mma_v.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                # once you've performed the gemm, 
                tOtO.store(tStO.load())


                mma_kv_pipeline_state.advance()
                load_kv_pipeline.consumer_release(mma_kv_pipeline_state)



if __name__ == "__main__":

    head_dims = 128
    batch_size = 4
    num_key_heads = 16
    group_size = 16
    qo_seq_len = 1024
    kv_seq_len = 1024
    B, S_qo, S_kv, H_kv, H_qo, D = batch_size, qo_seq_len, kv_seq_len, num_key_heads, num_key_heads * group_size, head_dims

    q_torch = torch.randn(B, H_qo, S_qo, D, dtype=torch.float16, device="cuda")
    k_torch = torch.randn(B, H_kv, S_kv, D, dtype=torch.float16, device="cuda")
    v_torch = torch.randn(B, H_kv, S_kv, D, dtype=torch.float16, device="cuda")
    o_torch = torch.randn(B, H_qo, S_qo, D, dtype=torch.float16, device="cuda")

    q = cutlass_torch.from_dlpack(q_torch, assumed_align=16)
    k = cutlass_torch.from_dlpack(k_torch, assumed_align=16)
    v = cutlass_torch.from_dlpack(v_torch, assumed_align=16)
    o = cutlass_torch.from_dlpack(o_torch, assumed_align=16)

    stream = cuda.CUstream()

    hopper_fa = HopperFA2(cta_tile=(64, 64, 128))
    compiled_kernel = cute.compile(hopper_fa, q, k, v, o, stream)
    compiled_kernel(q, k, v, o, stream)
    
