import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    return np.sqrt(np.sum((np.array(x)-np.array(y))**2))