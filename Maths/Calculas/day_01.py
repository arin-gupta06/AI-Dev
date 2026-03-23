import sympy as sp
import numpy as np
x = sp.Symbol('x')
f = sp.cos(x)
deri = sp.diff(f, x)
# print("The derivative of f(x) = sin(x) is:", deri)


#Gradient decent eg
a, b = sp.symbols('a, b')
f = a**2 + b**2
grad_a = sp.diff(f, a)
grad_b = sp.diff(f, b)
print("The function f(a, b) =", f)
print("Gradient with respect to a:", grad_a)
print("Gradient with respect to b:", grad_b)


# Gradient descent in Logical Regression
def gradient_descent(x, y, theta, learning_rate, iterations):
        m = len(y)
        for _ in range(iterations):
            predictions = np.dot(x, theta)
            errors = predictions - y
            gradient = (1/m) * np.dot(x.T, errors)
            theta -= learning_rate * gradient
        return theta

# Example usage
x = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([0, 1, 1])
theta = np.array([0.5, 0.5])
learning_rate = 0.01
iterations = 1000
final_theta = gradient_descent(x, y, theta, learning_rate, iterations)
print("Final parameters after gradient descent:", final_theta)