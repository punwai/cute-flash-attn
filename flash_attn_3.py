# 
# Writing a version of GQA flash attention 3 in CuTe DSL
# 

from typing import Type, Tuple
import cutlass
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import tcgen05, warpgroup
import cutlass.utils.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm90_utils
import cutlass.utils.hopper_helpers as sm100_utils
from cutlass import Int32

class HopperFusedMultiHeadAttentionForward:
    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool
    ):
        """
        :param mma_tiler: (M, N, K) shape of the MMA instruction.
        """
        self.qk_acc_dtype = qk_acc_dtype
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

    @cute.kernel
    def kernel(
        self,
        q: cutlass_torch.Tensor,
        k: cutlass_torch.Tensor,
        v: cutlass_torch.Tensor,
    ):
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        pass

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
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
        q = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d*h_r*h_k,1,((d * h_k, d)), stride_b_q)
        )
        # dim1: dimension
        # dim2: key group
        # dim3: index in a key group
        # dim4: sequence length
        # dim5: batch
        k = cute.make_layout(
            (s_k, d, ((h_r, h_k)), b),
            stride=(d * h_r * h_k, 1, ((0, d)), stride_b_kv)
        )

        # batch stride
        stride_b_qo = h_r * h_k * s_q * d

        # h_r is the number of queries in a group.

        # in order to 
        q = cute.make_tensor(q_iter, q)


        self.q_dtype = q.element_type

        cta_group = tcgen05.CtaGroup.ONE
        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler[:2],
        )

        # Allocate smem

        # based on the MMA tiler, make the smem layout for q and k.
        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )




        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.softmax_warpgroup_count
            ]
            corr_epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

