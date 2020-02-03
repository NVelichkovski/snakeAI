
class Status:
    DEAD = 1
    ACTIVE = 0


class Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Actions:
    LEFT = -1
    FORWARD = 0
    RIGHT = 1

    COORDINATES_OFFSET = {
        Direction.NORTH: {
            FORWARD: (-1, 0),
            RIGHT: (0, 1),
            LEFT: (0, -1)
        },
        Direction.EAST: {
            FORWARD: (0, 1),
            RIGHT: (1, 0),
            LEFT: (-1, 0)
        },
        Direction.SOUTH: {
            FORWARD: (1, 0),
            RIGHT: (0, -1),
            LEFT: (0, 1)
        },
        Direction.WEST: {
            FORWARD: (0, -1),
            RIGHT: (-1, 0),
            LEFT: (1, 0)
        }
    }


class Cell:
    # BGR FORMAT
    # BLOCK_CELL = (105, 105, 105)
    # EMPTY_CELL = (0, 0, 0)
    # SNAKE_BODY = (240, 128, 128)
    # SNAKE_HEAD = (128, 0, 0)
    # FOOD = (173, 255, 47)

    # RGB FORMAT
    BLOCK_CELL = (79, 79, 47)

    EMPTY_CELL = (0, 0, 0)  # IT HAS TO BE ZERO!
    # The matrix is initialized as a zero matrix with the shape of [width, height, color_channels]

    SNAKE_BODY = (128, 128, 240)
    SNAKE_HEAD = (0, 0, 128)
    FOOD = (47, 255, 173)

    CELL_REPRESENTATION = {BLOCK_CELL: ':', EMPTY_CELL: ' ', SNAKE_BODY: 'O', SNAKE_HEAD: '@', FOOD: 'G'}
