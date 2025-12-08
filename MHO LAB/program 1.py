import math
import matplotlib.pyplot as plt
#Define Objective function
# x = 0
# obj_func =  x**2 - 4*x + 5
# complex_function = math.sin(x**2/2) + 0.5*math.cos(2*x)
xmin = -5
xmax = 10
values = []
values_to_plot = []

for x in range(xmin, xmax + 1):
    obj_value = x**2 - 4*x + 5 #+ math.sin(x**2/2) + 0.5*math.cos(2*x)
    values.append((x, obj_value))
    values_to_plot.append(obj_value)
    print(f"x: {x}, Objective Function Value: {obj_value}")

print(values)
plt.plot(values_to_plot)
plt.show()

#Global Minimum
min_value = min(values, key=lambda item: item[1])
print(f"Global Minimum occurs at x = {min_value[0]} with value = {min_value[1]}")
