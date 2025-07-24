# 
# Gemm code utilizing the concept of worktiles.
# 

from typing import Tuple
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cuda.bindings.driver as cuda
import torch
import cutlass

WARP_SIZE = 32

class Gemm:
    def __init__(self, problem_shape: Tuple[int, int, int], cta_tile: Tuple[int, int, int]):
        self.problem_shape = problem_shape
        self.cta_tile = cta_tile
        self.pipeline_stages = 3
        self.buffer_align_bytes = 1024

    @cute.kernel
    def kernel(
        self,
        # global tma tensors
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        # tma atoms
        a_tma_atom: cute.CopyAtom,
        b_tma_atom: cute.CopyAtom,
        c_tma_atom: cute.CopyAtom,
        # shared storage
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        # mma
        tiled_mma: cute.TiledMma,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(a_tma_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(b_tma_atom)

        # initialize smem tensors
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA: cute.Tensor = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB: cute.Tensor = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # split global tensors
        gA = cute.zipped_divide(gA, cute.slice_(self.cta_tile, (None, 0, None)))[None, (bidx, None)]
        gB = cute.zipped_divide(gB, cute.slice_(self.cta_tile, (0, None, None)))[None, (bidy, None)]
        gC = cute.zipped_divide(gC, cute.slice_(self.cta_tile, (None, None, 0)))[None, (bidx, bidy)]
        k_tile_cnt = cute.size(gA, mode=[1])
        assert k_tile_cnt == cute.size(gB, mode=[1])

        # initialize pipelines
        mainloop_pipeline_array_ptr = storage.mainloop_mbar_ptr.data_ptr()

        # actual pipeline
        a_smem_unstaged_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_unstaged_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_unstaged_layout) \
            + cute.size_in_bytes(self.b_dtype, b_smem_unstaged_layout)
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.pipeline_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
        )

        is_producer = warp_idx == 0
        is_consumer = True

        # partition tma tensors
        cluster_shape_mnk = (1, 1, 1)
        cta_layout = cute.make_layout(cluster_shape_mnk)
        cta_layout = cute.make_layout(cute.slice_(cta_layout, (0, None, 0)).shape)
        cta_crd = (0,)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            a_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            gA,
        )
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            b_tma_atom,
            cta_crd,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            gB,
        )

        # partition smem for mma
        thr_mma = tiled_mma.get_slice(0)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)


        if is_producer:
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.pipeline_stages
            )
            for k_tile_idx in cutlass.range(k_tile_cnt, unroll=1):
                # what acquire does is to signal that you are readying this barrier up for the TMA.
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                cute.copy(
                    a_tma_atom,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                )
                cute.copy(
                    b_tma_atom,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state)
                )
                mainloop_producer_state.advance()

        if is_consumer:
            mainloop_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.pipeline_stages
            )
            for k_tile_idx in cutlass.range(k_tile_cnt, unroll=1):
                # wait for the tile to be loaded.
                mainloop_pipeline.consumer_wait(mainloop_consumer_state)
                cute



    @cute.jit
    def __call__(
        self, 
        gA: cute.Tensor, 
        gB: cute.Tensor, 
        gC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        a_smem_shape = (self.cta_tile[0], self.cta_tile[2])
        b_smem_shape = (self.cta_tile[1], self.cta_tile[2])
        c_smem_shape = (self.cta_tile[0], self.cta_tile[1])
        self.a_dtype = gA.element_type
        self.b_dtype = gB.element_type
        
        def get_staged_layout(smem_shape: Tuple[int, int], type: type[cutlass.Numeric], stages: int):
            smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
                type,
            )
            return cute.tile_to_shape(
                smem_layout_atom,
                cute.append(smem_shape, stages),
                order=(0, 1, 2)
            )

        # 1. smem layouts for pipeline stages
        a_smem_layout_staged = get_staged_layout(a_smem_shape, gA.element_type, self.pipeline_stages)
        b_smem_layout_staged = get_staged_layout(b_smem_shape, gB.element_type, self.pipeline_stages)
        c_smem_layout_staged = get_staged_layout(c_smem_shape, gC.element_type, self.pipeline_stages)

        # 2. tma atoms & tensors
        g2s_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        s2g_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        a_tma_atom, a_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            gA,
            a_smem_layout,
            (self.cta_tile[0], self.cta_tile[2]),
        )

        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        b_tma_atom, b_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s_op,
            gB,
            b_smem_layout,
            (self.cta_tile[1], self.cta_tile[2]),
        )

        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
        epi_tile = sm90_utils._sm90_compute_tile_shape_or_override(
            self.cta_tile, gC.element_type, is_cooperative=False
        )
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(gC.shape), epi_tile
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            s2g_op,
            gC,
            epi_smem_layout,
            c_cta_v_layout,
        )

        # 3. tiled mma
        mm = cute.nvgpu.warpgroup.OperandMajorMode.K
        self.atom_layout_mnk = (1, 1, 1)
        self.acc_dtype = cutlass.Float32
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            gA.element_type,
            gB.element_type,
            mm,
            mm,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile[1]),
        )


        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[gA.element_type, cute.cosize(a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[gB.element_type, cute.cosize(b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            mainloop_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.pipeline_stages * 2]

        self.shared_storage = SharedStorage

        # how are you going to split this?
        num_m_blocks = self.problem_shape[0] // self.cta_tile[0]
        num_n_blocks = self.problem_shape[1] // self.cta_tile[1]
        self.kernel(
            a_tma_tensor, b_tma_tensor, a_tma_atom, b_tma_atom, a_smem_layout_staged, b_smem_layout_staged, tiled_mma
        ).launch(
            grid=[num_m_blocks, num_n_blocks, 1],
            block=[4 * WARP_SIZE, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        print("done")


if __name__ == "__main__":
    gemm = Gemm((1024, 1024, 1024), (128, 128, 128))
    a_trch = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b_trch = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c_trch = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    a = cutlass_torch.from_dlpack(a_trch, assumed_align=16)
    b = cutlass_torch.from_dlpack(b_trch, assumed_align=16)
    c = cutlass_torch.from_dlpack(c_trch, assumed_align=16)

    stream = cuda.CUstream()

    compiled_kernel = cute.compile(gemm, a, b, c, stream)
    compiled_kernel(a, b, c, stream)