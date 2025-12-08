import math
import matplotlib.pyplot as plt
pi = math.pi

xmin = int(-pi/2)
xmax = int(pi/2)
values = []
values_to_plot = []

for x in range(xmin, xmax + 1):
    obj_value = math.sin(x**2/2) + 0.5*math.cos(2*x)
    values.append((x))
    values_to_plot.append(obj_value)
    print(f"x: {x}, Objective Function Value: {obj_value}")

plt.plot(values, values_to_plot)
plt.show()

#Global Minimum
min_value = min(values, key=lambda item: item[1])
print(f"Global Minimum occurs at x = {min_value[0]} with value = {min_value[1]}")