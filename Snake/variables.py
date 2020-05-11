
class Status:
    DEAD = 1
    ACTIVE = 0


class Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    COORDINATES_OFFSET = {
        NORTH: (-1, 0),
        EAST: (0, 1),
        SOUTH: (1, 0),
        WEST: (0, -1)
    }


class Cell:
    BLOCK_CELL = -1

    EMPTY_CELL = 0  # IT HAS TO BE ZERO!
    # The matrix is initialized as a zero matrix with the shape of [width, height, color_channels]

    FOOD = 1

    SNAKE_BODY = 2
    SNAKE_HEAD = 3

    CELL_DICT = {
        'BLOCK_CELL': BLOCK_CELL,
        'EMPTY_CELL': EMPTY_CELL,
        'SNAKE_BODY': SNAKE_BODY,
        'SNAKE_HEAD': SNAKE_HEAD,
        'FOOD': FOOD
    }

    CELL_REPRESENTATION = {
        BLOCK_CELL: ':',
        EMPTY_CELL: ' ',
        SNAKE_BODY: 'O',
        SNAKE_HEAD: '@',
        FOOD: 'F'}


class CellRenderEnc:
    # # BGR FORMAT
    # # The matrix is initialized as a zero matrix with the shape of [width, height, color_channels]
    #
    # BLOCK_CELL = (105, 105, 105)
    # EMPTY_CELL = (0, 0, 0) # IT HAS TO BE ZERO!
    # SNAKE_BODY = (240, 128, 128)
    # SNAKE_HEAD = (128, 0, 0)
    # FOOD = (173, 255, 47)

    # RGB FORMAT
    # The matrix is initialized as a zero matrix with the shape of [width, height, color_channels]

    BLOCK_CELL = (79, 79, 47)
    EMPTY_CELL = (0, 0, 0)  # IT HAS TO BE ZERO!

    OTHER_SNAKE_BODY = (186, 255, 179)
    OTHER_SNAKE_HEAD = (135, 50, 168)

    MY_SNAKE_BODY = (128, 128, 240)
    MY_SNAKE_HEAD = (0, 0, 128)

    FOOD = (47, 255, 173)

    CELL_DICT = {
        'BLOCK_CELL': BLOCK_CELL,
        'EMPTY_CELL': EMPTY_CELL,
        'SNAKE_BODY': OTHER_SNAKE_BODY,
        'SNAKE_HEAD': OTHER_SNAKE_HEAD,
        'FOOD': FOOD
    }

    CELL_REPRESENTATION = {
        BLOCK_CELL: ':',
        EMPTY_CELL: ' ',
        MY_SNAKE_BODY: 'o',
        MY_SNAKE_HEAD: '@',
        OTHER_SNAKE_BODY: 'o',
        OTHER_SNAKE_HEAD: 'G',
        FOOD: 'F'}
