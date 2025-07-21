from typing import Tuple
from flash_attn_3 import HopperFusedMultiHeadAttentionForward
import cutlass.torch as cutlass_torch
import cutlass
import torch
from torch.utils.dlpack import to_dlpack
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import cuda.bindings.driver as cuda


def create_cute_tensor(
    shape: Tuple[int, ...], 
    dtype, 
):
    f32_torch_tensor_full = torch.empty(shape, dtype=torch.float32).random_(
        -2 if dtype.is_float or dtype.signed else 0, 2
    ).to(dtype=torch.float32)
    cute_tensor = cutlass_torch.from_dlpack(to_dlpack(f32_torch_tensor_full), assumed_align=16)
    cute_tensor.element_type = dtype
    return f32_torch_tensor_full, cute_tensor

# q: (batch_size, qo_seq_len, num_qo_heads, head_dim)
# k: (batch_size, kv_seq_len, num_kv_heads, head_dim)
# v: (batch_size, kv_seq_len, num_kv_heads, head_dim)
# o: (batch_size, qo_seq_len, num_qo_heads, head_dim)
def torch_flash_attn_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    flash_attn_3 = HopperFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float16,
        pv_acc_dtype=cutlass.Float16,
        mma_tiler=(64, 64, 64),
        is_persistent=True,
    )
    o = torch.empty_like(q)
    problem_size = (
        cutlass.Int32(q.shape[0]),
        cutlass.Int32(q.shape[2]),
        cutlass.Int32(k.shape[2]),
        cutlass.Int32(q.shape[1]),
        cutlass.Int32(k.shape[1]),
        cutlass.Int32(q.shape[3]),
    )

    in_dtype = cutlass.Float16
    out_dtype = cutlass.Float16

    q: cutlass.Tensor = cutlass_torch.from_dlpack(q, assumed_align=16)
    k: cutlass.Tensor = cutlass_torch.from_dlpack(k, assumed_align=16)
    v: cutlass.Tensor = cutlass_torch.from_dlpack(v, assumed_align=16)
    o: cutlass.Tensor = cutlass_torch.from_dlpack(o, assumed_align=16)
    q.element_type = in_dtype
    k.element_type = in_dtype
    v.element_type = in_dtype
    o.element_type = out_dtype


    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_kernel = cute.compile(
        flash_attn_3,
        q.iterator,
        k.iterator,
        v.iterator,
        o.iterator,
        problem_size,
        stream
    )

    compiled_kernel(q.iterator, k.iterator, v.iterator, o.iterator, problem_size, stream)

    return o