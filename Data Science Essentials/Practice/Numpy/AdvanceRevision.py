import numpy as np
#Ques-1: Given 
# a 3×3 array, write code to compute:
# Sum of all elements
# Column-wise sum
# Row-wise sum

array_3D = np.arange(1,10).reshape(3,3)
print("Original array:\n", array_3D)
print("Sum of all elements:", np.sum(array_3D))
print("Column-wise sum:,", np.sum(array_3D, axis=0))
print("Row-wise sum:", np.sum(array_3D, axis=1))