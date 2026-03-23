import numpy as np
# Ques 1:
#creating a 3x3 matrix
# test_arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
# print("Original array:\n", test_arr)
# print("\n")
# # print(test_arr * 2)
# print("Transpose matrix: \n",test_arr.T)

#Ques 2:

# test_arr4d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print("Sum of Rows: ",np.sum(test_arr4d, axis = 1))
# print("Sum of Column: ",np.sum(test_arr4d, axis = 0))

#Ques 3:
#Normalisation -> (x-mean)/std
test_arr = np.array([1,2,3,4,5,6,7,8,9])
# print((test_arr - np.mean(test_arr))/np.std(test_arr))

# Normalisation scaling 0-1
# print((test_arr - np.min(test_arr))/(np.max(test_arr) - np.min(test_arr)))


# Generating the random array and finding the min and max value from the array
arr = np.random.randint(0, 10, (3,3))
print("Original matrix: \n",arr)
print("Minimum Value:", np.min(arr))
print("Maximum Value:", np.max(arr))