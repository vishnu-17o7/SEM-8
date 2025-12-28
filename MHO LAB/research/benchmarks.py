import numpy as np

def sphere(x):
    """f1: Sphere Function (Unimodal)"""
    return np.sum(x ** 2)

def rastrigin(x):
    """f2: Rastrigin Function (Highly Multimodal)"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """f3: Rosenbrock Function (Valley-Shaped)"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))

def ackley(x):
    """f4: Ackley Function (Many Local Minima)"""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum1/d)) - np.exp(sum2/d) + 20 + np.e

def schwefel(x):
    """f5: Schwefel Function (Deceptive)"""
    return 418.9829 * len(x) - sum(x[i] * np.sin(np.sqrt(abs(x[i])))
                                    for i in range(len(x)))

def griewank(x):
    """f6: Griewank Function (Many Regular Minima)"""
    sum_part = sum(x[i]**2 for i in range(len(x))) / 4000
    prod_part = np.prod([np.cos(x[i] / np.sqrt(i+1)) for i in range(len(x))])
    return sum_part - prod_part + 1

BENCHMARKS = {
    'Sphere': (sphere, np.array([-5.12, 5.12])),
    'Rastrigin': (rastrigin, np.array([-5.12, 5.12])),
    'Rosenbrock': (rosenbrock, np.array([-2.048, 2.048])),
    'Ackley': (ackley, np.array([-32.768, 32.768])),
    'Schwefel': (schwefel, np.array([-500, 500])),
    'Griewank': (griewank, np.array([-600, 600]))
}
