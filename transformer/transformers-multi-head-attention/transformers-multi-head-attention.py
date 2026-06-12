import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    d_k = W_q.shape[-1]//num_heads
    
    Q_hat = Q @ W_q # (batch, seq, d_k)  
    K_hat = K @ W_k # (batch, seq, d_k) 
    V_hat = V @ W_v # (batch, seq, d_k) 
    
    Q_hat = Q_hat.reshape(Q.shape[0], Q_hat.shape[1], num_heads, d_k) # (batch, seq, num_heads, d_k) 
    K_hat = K_hat.reshape(K.shape[0], K_hat.shape[1], num_heads, d_k) # (batch, seq, num_heads, d_k)
    V_hat = V_hat.reshape(V.shape[0], V_hat.shape[1], num_heads, d_k) # (batch, seq, num_heads, d_k)

    Q_hat = Q_hat.transpose(0 ,2 ,1 ,-1)
    K_hat = K_hat.transpose(0 ,2 ,1 ,-1)
    V_hat = V_hat.transpose(0 ,2 ,1 ,-1)
    
    S = Q_hat @ K_hat.transpose(0,1,3,2)
    S_sq = S/np.sqrt(d_k)
    
    W = softmax(S_sq,axis=-1)
    O = W @ V_hat

    O = O.transpose(0, 2, 1, 3)  # (batch, seq, num_heads, d_k)
    return O.reshape(O.shape[0],O.shape[1],d_k*num_heads) @ W_o