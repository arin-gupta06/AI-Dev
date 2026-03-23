import numpy as np

arr = np.array([1,3,4,5])
# print(arr)

#2d array:
arr_2d = np.array([[1,2,3],[2,3,4],[7,8,9]])
# print(arr_2d)

#3d array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(arr_3d)

#Operations on numpy arrays

#zeros
zeros = np.zeros((3,4))
# print(zeros)

#ones
ones = np.ones((3,3))
# print(ones)

# range
array_range = np.arange(0, 10, 2)
# print(array_range)

# array linspace
array_linsp = np.linspace(0, 1, 10)
# print(array_linsp)

# identity matrix
# print(np.eye(3))

# random arrays
print(np.random.rand(3,3))

