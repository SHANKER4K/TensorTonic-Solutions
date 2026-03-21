def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def f(x):
        return a*x**2+b*x+c
    def df(x):
        return 2*a*x+b
    for i in range(steps):
        pred = f(x0)
        x0 -= lr*df(x0)
    return x0