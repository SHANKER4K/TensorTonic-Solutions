import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    _ ,counts = np.unique(y,return_counts=True)
    p = counts/len(y)
    return np.sum(-p*np.log2(p))