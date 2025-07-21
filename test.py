# Test Harness.
import cutlass
from flash_attn_3 import HopperFusedMultiHeadAttentionForward
from torch_harness import *
import torch

def test_flash_attn_3_correctness(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    reference_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )

    flash_attn_3 = HopperFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.BFloat16,
        mma_tiler=(1, 1, 1),
        is_persistent=True,
    )

    # flash_attn_3_output = flash_attn_3(q, k, v)
    
    


if __name__ == "__main__":
    test_flash_attn_3_correctness(
        batch_size=1,
        seq_len=512,
        head_dim=64,
        num_q_heads=32,
        num_kv_heads=8,
    )