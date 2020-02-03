from math import sqrt
from typing import Tuple

from environment.variables import Actions


def new_position(position, direction, action, matrix):
    i_offset, j_offset = Actions.COORDINATES_OFFSET[direction][action]
    return (position[0] + i_offset) % matrix.shape[0], (position[1] + j_offset) % matrix.shape[1]


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
