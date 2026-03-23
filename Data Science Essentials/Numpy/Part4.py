#Broadcasting

import numpy as np
# Scalar broadcasting
arr = np.array([1, 2, 3])
# print(arr + 10)


#Array broadcasting

matrix = np.array([[1, 3, 5], [4, 2, 6]])
vector = np.array([1, 0 , 1])
# print(matrix + vector)

# # Aggregate functions
# print("Sum: ", np.sum(matrix))
# print("Mean: ", np.mean(matrix))
# print("Median: ", np.median(matrix))
# print("Standard Deviation: ", np.std(matrix))
# print("Variance: ", np.var(matrix))

#Filtering
evens = matrix[matrix % 2 == 0]
print(evens)