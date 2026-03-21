def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    L = max_len
    if max_len == None:
        L = max(len(seq) for seq in seqs)
    arr = np.full((len(seqs),L),pad_value)
    for i in range(len(arr)):
        s = seqs[i]
        for k in range(min(L,len(s))):
            arr[i,k] = s[k]
    return arr
        