import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,3,4,5,6,7,8,10])
y = np.array([4,3,7,7,8,10,8,11])

l_rate = 0.01 # Learning rate
steps = 4000    # number of iterations ( steps )

m = 0 # initial slope
b = 0 # initial intercept

n = float(len(x))

m_array = []
b_array = []

# Start Gradient Descent
for step in range(steps) :
    y_pred = m * x + b

    # Derivative of the cost function w.r.t slope (m)
    m_der  = (-1/n) * sum( (y - y_pred) * x)
    # Derivative of the cost function w.r.t intercept (b)    
    b_der  = (-1/n) * sum( y-y_pred )
    
    # move m
    m = m -  l_rate * m_der
    b = b -  l_rate * b_der
    
    # gather the slope and intercept in an array to plot later 
    m_array.append(m)
    b_array.append(b)
    
print (" optimim slope(m) = ", m)
print ( "optimum intercept (m) = ", b)
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

slope_values     = np.arange(start=0,stop=2,step=0.05)
intercept_values = np.arange(start=0,stop=2,step=0.05)

n = len(y)

cost_function = []

for index, slope in enumerate(slope_values) : 
    cost = 0
    for i in range(n):
        cost = cost + (1/(2*n)) * ( (y[i] - slope_values[index] * x[i] - intercept_values[index]) ** 2 )
    cost_function.append(cost)

slope_values_new     = m_array
intercept_values_new = b_array

cost_function_new = []
for index, slope in enumerate(slope_values_new) : 
    cost = 0
    for i in range(n):
        cost = cost + (1/(2*n)) * ( (y[i] - slope_values_new[index] * x[i] - intercept_values_new[index]) ** 2 )
    cost_function_new.append(cost)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(slope_values, intercept_values, cost_function,marker='o')
ax.scatter(slope_values_new, intercept_values_new, cost_function_new,marker='o')

plt.show()