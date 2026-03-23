import numpy as np


arrays = np.arange(1,10)
# print(arrays.shape)
# print(arrays.reshape(3,3))
# print(np.expand_dims(arrays, axis = 2).shape)



#muliplications

nums_01  =  np.arange(1, 51).reshape(5, 10)
nums_02  =  np.arange(1, 101).reshape(10, 10)

mult = nums_01 @ nums_02
# print(mult)


#Norms

data = np.arange(1, 1001).reshape(10, 100)

norm_01 = np.linalg.norm(data, 1)

# print(mean)

# root square mean value 

norm_02 = np.linalg.norm(data, 2)

# print(mean_sq)

# max value

norm_03 = np.linalg.norm(data, np.inf)

# print(max_val)

# let's see how max value is deviated from the mean value
devit = norm_01 - norm_03
# print(devit)








# Mini project- Recommendation system

items = np.array([
    [1, 0, 1],   # Movie A
    [1, 1, 0],   # Movie B
    [0, 1, 1]    # Movie C
])

# Similarity

sim_A_B = items[0] @ items[1]
sim_A_C = items[0] @ items[2]
sim_B_C = items[1] @ items[2]

# print(sim_A_B)
# print(sim_A_C)
# print(sim_B_C)

# norm vectors of items

norm_A = np.linalg.norm(items[0])
norm_B = np.linalg.norm(items[1])
norm_C = np.linalg.norm(items[2])

# print(norm_A)
# print(norm_B)
# print(norm_C)


# Cosine similarity (final similarity)

cos_sim_A_B = sim_A_B / (norm_A * norm_B)
cos_sim_A_C = sim_A_C / (norm_A * norm_C)
cos_sim_B_C = sim_B_C / (norm_B * norm_C)

# print(cos_sim_A_B)
# print(cos_sim_A_C)
# print(cos_sim_B_C)



# Mini Project - 02 User-User Similarity

import numpy as np

users = np.array([
    [5, 0, 3, 0, 2],
    [4, 0, 3, 1, 2],
    [0, 5, 0, 4, 0],
    [5, 0, 4, 0, 1],
    [0, 4, 0, 5, 0]
])


# Similarity between users 
sim_user1_and_user2 = users[0] @ users[1]
sim_user1_and_user3 = users[0] @ users[2]
sim_user1_and_user4 = users[0] @ users[3]
sim_user1_and_user5 = users[0] @ users[4]

# Similarity pattern

sim_1_2 = sim_user1_and_user2/(np.linalg.norm(users[0]) * (np.linalg.norm(users[1])))
sim_1_3 = sim_user1_and_user3/(np.linalg.norm(users[0]) * (np.linalg.norm(users[2])))
sim_1_4 = sim_user1_and_user4/(np.linalg.norm(users[0]) * (np.linalg.norm(users[3])))
sim_1_5 = sim_user1_and_user5/(np.linalg.norm(users[0]) * (np.linalg.norm(users[4])))

print(sim_1_2)
print(sim_1_3)
print(sim_1_4)
print(sim_1_5)