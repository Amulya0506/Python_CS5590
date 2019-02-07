# import numpy package
import numpy as np
# generate 15 random integers in the range 0-20
vectorList = np.random.randint(0,20,15)
print(vectorList)
# bincount counts number of occurrences of each value in array of positive integers.
counts = np.bincount(vectorList)
# argmax returns the indices of the maximum values
print("Most frequent item in the list is: %d" %(np.argmax(counts)))
