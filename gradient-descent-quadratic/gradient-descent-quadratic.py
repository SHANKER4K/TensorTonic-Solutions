def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def df(x):
        return 2*a*x+b
    for i in range(steps):
        x0 -= lr*df(x0)
    return x0