from flash_attn_3 import HopperFusedMultiHeadAttentionForward
import cutlass
import torch

# q: (batch_size, num_q_heads, seq_len, head_dim)
def torch_flash_attn_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    flash_attn_3 = HopperFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.BFloat16,
        mma_tiler=(1, 1, 1),
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
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    # q = cutlass_torch.Tensor(q)
    # flash_attn_3_output = flash_attn_3(q, k, v, o, problem_size)

    # return flash_attn_3_output