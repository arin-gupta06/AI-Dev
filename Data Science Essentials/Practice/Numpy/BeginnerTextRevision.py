import numpy as np
#Ques-1: Write a program to create a 1D NumPy array of integers from 10 to 20 and print its datatype.

array = np.arange(10,21)
# print(type(array))

#Ques-2: Create a 3×3 ndarray filled with zeros and then change the datatype to float32.
zeros = np.zeros([3,3])
zeros = zeros.astype(np.float32)
# print(zeros.dtype)

#Ques-3: Write code to create a 2×4 array using np.arange() and reshape it appropriately.
nums = np.arange(0,8)
nums = nums.reshape(2,4)
# print(nums)

#Ques-4: Create an ndarray from the list [1, 2, 3, 4, 5] and multiply all elements by 5 using vectorized computation.
numbers = np.array([1, 2, 3, 4, 5])
# print("Original array:", numbers)
numbers *= 5
# print("After multiplying by 5:", numbers)

#Ques-5: Write a program to generate a 4×4 identity matrix.
Indentity_matrix = np.eye(5)
print(Indentity_matrix)