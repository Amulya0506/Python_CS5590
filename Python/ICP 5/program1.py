import numpy as np
import matplotlib.pyplot as plt
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
x_mean = np.mean(x)
y_mean = np.mean(y)
# change in x
numerator = sum((x-x_mean)*(y-y_mean))
# change in y
denominator = sum(np.power((x-x_mean),2))
slope = numerator/denominator
print(slope)
# regression coefficients
# y- intercept
intercept = y_mean-(slope*x_mean)
print(intercept)
regY = (slope*x)+intercept
plt.plot(x,regY)
plt.scatter(x,regY)
plt.plot(x,y)
plt.show()

