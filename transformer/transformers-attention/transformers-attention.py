import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    S = Q @ K.transpose(-2,-1) # n×m
    S_sq = S/math.sqrt(K.shape[-1]) # n×m
    
    W = F.softmax(S_sq,dim=-1)
    O = W @ V # m×d_v
    
    return O