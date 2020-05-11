from math import sqrt
from typing import Tuple

from Snake.variables import Direction


def new_position(position, new_direction, matrix):
    i_offset, j_offset = Direction.COORDINATES_OFFSET[new_direction]
    return (position[0] + i_offset) % matrix.shape[0], (position[1] + j_offset) % matrix.shape[1]


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
