import numpy as np
import random

def g(x):
    return np.sin((x**2)/2) + 0.5 * (np.cos(2*x))

def stochastic_Hill_Climbing():
    xmin = -5
    xmax = 5
    delta = 0.5
    current_sol = random.uniform(xmin,xmax)

    print("Starting point : X = ",current_sol,"with g(x) value = ",g(current_sol))

    for _ in range(100):
        neighbour =  current_sol + random.uniform(-delta,delta)
        # print(neighbour,current_sol)
        if(g(current_sol) > g(neighbour)):
            print("Found the current best solution at iteration:",_)
            current_sol = neighbour

    print("After 100 iteration, the X is",current_sol,"with g(X) = ",g(current_sol))


stochastic_Hill_Climbing()

