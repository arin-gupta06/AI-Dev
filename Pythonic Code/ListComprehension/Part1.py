numbers = [1, 2, 3, 4, 5]

even_numbers = [x for x in numbers if x % 2 == 0]
# print(even_numbers)  # Output: [2, 4]

squared_numbers = [x**2 for x in numbers]
# print(squared_numbers)  # Output: [1, 4, 9, 16, 25]


add = lambda x, y: x + y
result = add(3, 5)

# print(result)   # Output: 8

sqaure = map(lambda y: y**2, numbers)
print(list(sqaure))  # Output: 16
