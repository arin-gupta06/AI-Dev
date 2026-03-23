#Ques-1: Create two 1D arrays:
#arr1 = [1,2,3,4]
#arr2 = [5,6,7,8]
#Perform element-wise addition, subtraction, and multiplication.

import numpy as np

arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
# print("Addition:", arr1 + arr2)
# print("Subtraction:", arr1 - arr2)
# print("Multiplication:", arr1 * arr2)

#Ques-3: Write code to convert an integer array into a float array without modifying the original array.
array_int = np.array([2,4,6,8])
array_float = array_int.astype(np.float32)
# print("Original array:", array_int)
# print("Float array:", array_float)

#Ques-4: Create a 2×2 array and compute its square using vectorized computation.
array_2D = np.array([[1,2],[4,5]])
squared_array = array_2D ** 2
# print("Original array:\n", array_2D)
# print("Squared array:\n", squared_array)

#Ques-5: 
# Given
# arr = np.array([[10,20,30],
#                 [40,50,60]])
# Write code to extract:
# The element 50
# The first row
# The second column

arr = np.array([[10,20,30],[40,50,60]])
# print("the resulted :", arr[1,1])

#Ques-6: Write code to slice rows from index 1 to 5 from a 1D array of size 10. Explain what happens if the end index exceeds array size.
array_to_be_sliced = np.arange(1,10)
# print("Original array:", array_to_be_sliced)
# print("Sliced array:", array_to_be_sliced[:5])

#Ques-7: Extract the last two columns of a 3×4 array using slicing.
array_3d = np.arange(1,13)
array_3d = array_3d.reshape(3,4)
# print("Original array:\n", array_3d)
# print("Last two columns:\n", array_3d[:, 2:])


#Ques-8: Write code to reverse a 1D array using slicing.
array = np.array([1,2,3,4,5])
reversed_array = array[::-1]
print("Original array:", array)
print("Reversed array:", reversed_array)
