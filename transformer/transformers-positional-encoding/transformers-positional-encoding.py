import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    res = np.zeros((seq_length,d_model),dtype=np.float32)
    pos = np.arange(seq_length).reshape(-1, 1)
    
    i = np.arange(0,d_model,2)
    div = np.exp((i/d_model)*-np.log(10000))
    
    res[:,0::2] = np.sin(pos*div)
    res[:,1::2] = np.cos(pos*div)
    
    return res.round(4)