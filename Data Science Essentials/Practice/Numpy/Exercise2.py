# Generating random dataset and filtering values based on conditions
import numpy as np
matrix = np.random.randint(1, 1001, (6,6))
print("Original Dataset: \n",matrix)

new_dataset = matrix[np.sqrt(matrix) == np.sqrt(matrix).astype(int)]
print("New Dataset: \n", new_dataset)