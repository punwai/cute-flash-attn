# 
# Writing a version of GQA flash attention 3 in CuTe DSL
# 

import math
from typing import Type, Tuple
import cutlass
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import tcgen05, warpgroup
import cutlass.utils.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32

class HopperFusedMultiHeadAttentionForward:
    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool
    ):
        """
        :param mma_tiler: (M, N, K) shape of the MMA instruction.
        """
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        # (M, N, K) for the attention is equivalent to (q_s_blk, k_s_blk, d_blk)
        self.mma_tiler = mma_tiler
        # For the CTA size, we will load in 2 q blocks and 1 k block.
        self.cta_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler # (q_s_blk, k_s_blk, d_blk)
        self.pv_mma_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1]) # (q_s_blk, d_blk, k_s_bl)
        self.is_persistent = is_persistent

        self.mma_warp_groups = math.prod(self.qk_mma_tiler)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = sm90_utils.SMEM_CAPACITY["sm90"]

        # Warp Specialization IDs
        self.load_warp_id = 1
        self.mma_warp_id = 2


    def make_and_init_load_q_pipeline(
        self, 
        load_q_mbar_ptr: cute.Pointer,
        cta_layout_vmnk: cute.Layout,
    ):
        load_q_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        load_q_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        return pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stage,
            producer_group=load_q_producer_group,
            consumer_group=load_q_consumer_group,
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=load_q_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk
        )

    # def make_


    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        q: cutlass_torch.Tensor,
        tma_atom_q: cute.Atom,
        k: cutlass_torch.Tensor,
        tma_atom_k: cute.Atom,
        v: cutlass_torch.Tensor,
        tma_atom_v: cute.Atom,
        o: cutlass_torch.Tensor,
        tma_atom_o: cute.Atom,
        qk_thr_mma: cute.TiledMma,
        pv_thr_mma: cute.TiledMma,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////
        #  Prefetch Tma descriptor 
        # /////////////////////////////

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # ///////////////////////////////
        #  Allocate and initialize smem  
        # ///////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_pipeline = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk
        )

        # ///////////////////////////////
        #  Allocate and initialize smem
        # ///////////////////////////////

        sQ: cute.Tensor = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK: cute.Tensor = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV_ptr: cute.Pointer = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV: cute.Tensor = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        sO: cute.Tensor = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )

        tidx, _, _ = cute.arch.thread_idx()

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        qk_thr_mma = qk_tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        pv_thr_mma = pv_tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        # qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm

        print("qk_thr_mma", qk_thr_mma)

        tSrQ = qk_thr_mma.partition_A(sQ)
        tSrK = qk_thr_mma.partition_B(sK)
        tOrV = pv_thr_mma.partition_B(sV)

        print("tSrQ", tSrQ)
        print("tSrK", tSrK)
        print("tOrV", tOrV)

        cute.print_tensor(tSrQ)

        tCsQ = qk_thr_mma.partition_A(sQ)
        tCsK = qk_thr_mma.partition_B(sK)
        tCsV = pv_thr_mma.partition_B(sV)

        tCrQ = qk_tiled_mma.make_fragment_A(tCsQ)
        tCrK = qk_tiled_mma.make_fragment_B(tCsK)
        tCrV = pv_tiled_mma.make_fragment_B(tCsV)

        # CuTe library for abstracting away important parts
        # of the system.

        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)

        pv_acc_shape = pv_thr_mma.partition_shape_C(
            (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
        )
        tStS0 = cute.make_tensor(tStS.iterator, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator, tStS.layout)

        # ///////////////////////////////
        # Tile Global Tensors
        # ///////////////////////////////
        
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        
        # q: (b, s_q, h_q, d)
        tile_coord_mnkl = (bidx, bidy, None, bidz)
        # how would you do this? I would load each 

        gQ = cute.local_tile(
            q.iterator, 
            qk_tiled_mma, # TODO: please replace this with CTA tiler, do not just use the qk_tiled_mma to 
        )

        # ///////////////////////////////
        # Prefetch Q & K + V
        # ///////////////////////////////
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        # ///////////////////////////////
        # k_tile_cnt = cute.size(gA_mkl, mode=[2])
        # prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)
        # q_producer_state = pipeline.make_pipeline_state(
        #     pipeline.PipelineUserType.Producer, self.q_stage
        # )
        # if warp_idx == 0:
        #     for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        # tile shape to perform the copy with.
        smem_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors"""
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tma_tile_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        stream: cuda.CUstream,
    ):
        """
        :param q_iter: Query tensor pointer
        :param k_iter: Key tensor pointer
        :param v_iter: Value tensor pointer
        :param o_iter: Output tensor pointer
        :param problem_size: Problem size with shape [b, s_q, s_k, q, k, d].
        qk_tiled_mma: cute.TiledMma,
        """
        b, s_q, s_k, h_q, h_k, d = problem_size
        # h_k is the number of heads in q.
        h_r = h_q // h_k
        # h_k is the number of groups
        # h_r is the number of queries per group
        stride_b_q = h_r * h_k * s_q * d
        stride_b_kv = h_k * s_k * d 

        # dim1: dimension
        # dim2: key group
        # dim3: index in a key group
        # dim4: sequence length
        # dim5: batch
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d*h_r*h_k,1,((d * h_k, d), stride_b_q)),
        )
        q = cute.make_tensor(q_iter, q_layout)

        # dim1: dimension
        # dim2: key group
        # dim3: index in a key group
        # dim4: sequence length
        # dim5: batch
        k_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b)),
            stride=(d * h_r * h_k, 1, ((0, d), stride_b_kv)),
        )
        k = cute.make_tensor(k_iter, k_layout)

        v_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b)),
            stride=(d * h_k, 1, ((0, d), stride_b_kv)),
        )
        v = cute.make_tensor(v_iter, v_layout)

        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d * h_r * h_k, 1, ((d * h_k, d), stride_b_q)),
        )
        o = cute.make_tensor(o_iter, o_layout)
        
        # batch stride
        stride_b_qo = h_r * h_k * s_q * d

        self.qk_mma_tiler = self.mma_tiler
        self.pv_mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[2],
            self.mma_tiler[1],
        )


        self.q_major_mode = utils.LayoutEnum.from_tensor(q).sm90_mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).sm90_mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).sm90_mma_major_mode()
        self.o_major_mode = utils.LayoutEnum.from_tensor(o).sm90_mma_major_mode()

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            (1, 1, 1),
            self.mma_tiler[:2],
        )
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            self.o_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            (1, 1, 1),
            self.pv_mma_tiler[:2]
        )


        # Allocate smem
        self.q_stage = 2
        self.kv_stage = 4
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2

        # /////////////////////////////////////////////////////////
        # based on the MMA tiler, make the smem layout for q and k.
        # /////////////////////////////////////////////////////////

        q_is_k_major = (
            utils.LayoutEnum.from_tensor(q).sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        k_is_k_major = (
            utils.LayoutEnum.from_tensor(k).sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )

        q_major_mode_size = self.qk_mma_tiler[2 if q_is_k_major else 0]
        q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                # it says which major mode the tensor you're transporting it.
                utils.LayoutEnum.from_tensor(q),
                self.q_dtype,
                q_major_mode_size,
            ),
            self.q_dtype,
        )
        q_smem_shape = cute.slice_(self.mma_tiler, (None, 0, None))
        self.q_smem_layout_staged = cute.tile_to_shape(
            q_smem_layout_atom,
            cute.append(q_smem_shape, self.q_stage),
            order=(0, 1, 2) if q_is_k_major else (1, 0, 2),
        )

        k_major_mode_size = self.qk_mma_tiler[1 if k_is_k_major else 0]
        k_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.from_tensor(k),
                self.k_dtype,
                k_major_mode_size,
            ),
            self.k_dtype,
        )
        k_smem_shape = cute.slice_(self.mma_tiler, (0, None, None))
        self.k_smem_layout_staged = cute.tile_to_shape(
            k_smem_layout_atom,
            cute.append(k_smem_shape, self.kv_stage),
            order=(0, 1, 2) if k_is_k_major else (1, 0, 2),
        )

        v_major_mode_size = self.qk_mma_tiler[1 if k_is_k_major else 0]
        v_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.from_tensor(v),
                self.v_dtype,
                v_major_mode_size,
            ),
            self.v_dtype,
        )
        v_smem_shape = cute.slice_(self.mma_tiler, (0, None, None))
        self.v_smem_layout_staged = cute.tile_to_shape(
            v_smem_layout_atom,
            cute.append(v_smem_shape, self.kv_stage),
            order=(0, 1, 2) if k_is_k_major else (1, 0, 2),
        )

        o_major_mode_size = self.qk_mma_tiler[2 if k_is_k_major else 0]
        o_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.from_tensor(o),
                self.v_dtype,
                o_major_mode_size,
            ),
            self.v_dtype,
        )
        o_smem_shape = cute.slice_(self.mma_tiler, (None, 0, None))
        self.o_smem_layout_staged = cute.tile_to_shape(
            o_smem_layout_atom,
            cute.append(o_smem_shape, self.epi_stage),
            order=(0, 1, 2) if k_is_k_major else (1, 0, 2),
        )

        
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.softmax_corr_stage * 2]
            # s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
            #     cutlass.Int64, self.softmax_warpgroup_count
            # ]
            # corr_epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_stage * 2]
            # mma_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_corr_stage * 2]
            # tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # Tmem holding buffer
            # tmem_holding_buf: cutlass.Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(self.o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            q,
            self.q_smem_layout_staged,
            (self.qk_mma_tiler[0], self.qk_mma_tiler[2]),
        )

        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            k,
            self.k_smem_layout_staged,
            (self.qk_mma_tiler[1], self.qk_mma_tiler[2]),
        )

        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            v,
            self.v_smem_layout_staged,
            (self.pv_mma_tiler[1], self.pv_mma_tiler[2]),
        )

        tma_atom_o, tma_tensor_o = self._make_tma_atoms_and_tensors(
            o,
            self.o_smem_layout_staged,
            (self.pv_mma_tiler[0], self.pv_mma_tiler[2]),
        )
        
        self.threads_per_cta = 1024

        q_smem_layout = cute.select(self.q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(self.k_smem_layout_staged, mode=[0, 1, 2])

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        self.cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

        # grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mnk)
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_tensor_q,
            tma_atom_q,
            tma_tensor_k,
            tma_atom_k,
            tma_tensor_v,
            tma_atom_v,
            tma_tensor_o,
            tma_atom_o,
            qk_tiled_mma,
            pv_tiled_mma,
            self.q_smem_layout_staged,
            self.k_smem_layout_staged,
            self.o_smem_layout_staged,
            self.v_smem_layout_staged,
            self.cta_layout_vmnk,
        ).launch(
            # grid=grid,
            grid = [1, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.utils.pipeline.PipelineAsync,
        pipeline_v: cutlass.utils.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
    ):
        pass
    
    # for the mma parts to perform.
    def mma(
        self,
    ):
        pass






