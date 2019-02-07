# importing NumPy package
import numpy as np

# Creating 10x10 array with random values and looping each row
for n in np.random.random((10, 10)):
    # printing each row
    print(n)
    print("Min & Max:  ")
    # finding min and max values in each row
    print((min(n), max(n)))
