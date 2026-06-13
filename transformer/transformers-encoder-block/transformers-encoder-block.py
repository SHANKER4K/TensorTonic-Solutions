import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mean = np.mean(x,axis=-1,keepdims=True)
    var = np.var(x,axis=-1,keepdims=True)

    res = gamma * (x-mean)/(np.sqrt(var+eps)) + beta
    return res

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor , d_k) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    S = Q @ K.transpose(0,1,3,2)
    S_sq = S/np.sqrt(d_k)
    
    W = softmax(S_sq,axis=-1)
    O = W @ V
    return O

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model//num_heads
    
    Q_hat = Q @ W_q
    K_hat = K @ W_k
    V_hat = V @ W_v

    Q_hat = Q_hat.reshape(batch_size,seq_len,num_heads,d_k)
    K_hat = K_hat.reshape(batch_size,seq_len,num_heads,d_k)
    V_hat = V_hat.reshape(batch_size,seq_len,num_heads,d_k)
    
    Q_hat = Q_hat.transpose(0,2,1,-1)
    K_hat = K_hat.transpose(0,2,1,-1)
    V_hat = V_hat.transpose(0,2,1,-1)
    
    O = scaled_dot_product_attention(Q_hat,K_hat,V_hat ,d_k)
    
    O = O.transpose(0,2,1,-1)
    O = O.reshape(batch_size,seq_len,num_heads*d_k)
    
    return O @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    p = x @ W1+b1
    return np.maximum(0,p) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    
    att = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    
    x = layer_norm(x + att,gamma1,beta1)
    
    ff = feed_forward(x,W1,b1,W2,b2)
    
    x = layer_norm(x + ff,gamma2,beta2)
    
    return x