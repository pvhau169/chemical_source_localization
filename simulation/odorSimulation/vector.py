import math
import numpy as np


def magnitude(vector):
    x, y = vector
    return math.sqrt((x ** 2) + (y ** 2))


def dot(a, b):
    return sum(i * j for i, j in zip(a, b))


def distance(a, b):
    c = np.array(a) - np.array(b)
    return magnitude(c)


def angle_between(a, b):
    angle = math.degrees(math.acos(dot(a, b) / (magnitude(a) * magnitude(b))))
    return angle/180 * np.pi


def limit_magnitude(vector, max_magnitude, min_magnitude=0.0):
    mag = magnitude(vector)
    if mag > max_magnitude:
        normalizing_factor = max_magnitude / (mag + (mag == 0))
    elif mag < min_magnitude:
        normalizing_factor = min_magnitude / (mag + (mag == 0))
    else:
        return vector

    return np.array([value * normalizing_factor for value in vector])


def rotate(vector, angle):
    vector = np.array(vector)
    matrix = [[math.cos(angle), -math.sin(angle)],
              [math.sin(angle), math.cos(angle)]]
    return np.matmul(matrix, vector.T)
