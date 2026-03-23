#Numpy array operations
import numpy as np

# Arithmetic Operations
test_arr1 = np.array([1,4,6])
test_arr2 = np.array([3,5,7])

# print(test_arr1 + test_arr2)
# print(test_arr1 - test_arr2)
# print(test_arr1 * test_arr2)
# print(test_arr1 / test_arr2)
# print(test_arr1 % test_arr2)
# print(test_arr1 % 2)
# print(test_arr2 * 2)

# Universal functions
# print(np.exp(test_arr1))
# print(np.square(test_arr1))


# Aggregate Functions
test_arr = np.array([[1,2,3],[4,5,6]])
# print(np.sum(test_arr))
# print(np.sum(test_arr, axis = 1))
# print(np.sum(test_arr, axis = 0))
# print(np.mean(test_arr))
print(np.std(test_arr))