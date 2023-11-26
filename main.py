import numpy as np
import time

def generate_random_points(seed, num_points):
    np.random.seed(seed % (2**32 - 1))
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    return x, y

def custom_function(x, y, coefficients):
    return coefficients[0] * np.exp(coefficients[1] * x) - y

def least_squares_fit(x, y, degree):
    A = np.vander(x, degree + 1)
    coefficients = np.linalg.inv(A.T @ A) @ A.T @ y
    return coefficients


seed_value = int(round(time.time() * 1000))
num_points = 10
x_points, y_points = generate_random_points(seed_value, num_points)
degree_of_fit = 2
coefficients = least_squares_fit(x_points, y_points, degree_of_fit)

print("拟合函数的系数:", coefficients)
