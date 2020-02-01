import numpy as np

class Cell:
    # BGR FORMAT
    BLOCK_CELL = (105, 105, 105)
    EMPTY_CELL = (0, 0, 0)
    SNAKE_BODY = (240, 128, 128)
    SNAKE_HEAD = (128, 0, 0)
    FOOD = (173, 255, 47)

    BLOCK_CELL = (105, 105, 105)
    EMPTY_CELL = (0, 0, 0)
    SNAKE_BODY = (128, 128, 240)
    SNAKE_HEAD = (0, 0, 128)
    FOOD = (47, 255, 173)

    CELL_REPRESENTATION = {BLOCK_CELL: ':', EMPTY_CELL: ' ', SNAKE_BODY: 'O', SNAKE_HEAD: '@', FOOD: 'G'}

class Environment:

    def __init__(self, width=0, height=0, max_num_agents=0, with_boundaries=True):
        if width < 5 or height < width:
            raise AttributeError(
                """
                    Environment with his dimensions cannot be constructed.
                    width > 5
                    height >= width
                    max_num_agents > 1
                """
            )
        self.width = width
        self.height = height
        self.max_num_agents = max_num_agents
        self.num_agents = self.max_num_agents if self.max_num_agents * 2 < self.width + 2 else (self.width - 2) // 2
        self.matrix = None
        self.with_boundaries = with_boundaries

        self.food = []
        self.occupied_cells = {}

        # TODO: Create Agent class and add it to the self.snakes
        self.snakes = {

        }

    def reset(self):
        boundaries_offset = 2 if self.with_boundaries else 0
        self.matrix = np.zeros((self.width + boundaries_offset, self.height + boundaries_offset, 3), dtype=np.uint8)
        for i in range(2, 2 + self.num_agents*2, 2):
            self.matrix[1][i] = Cell.SNAKE_BODY
            self.matrix[2][i] = Cell.SNAKE_HEAD
            self.occupied_cells[(1, i)] = Cell.SNAKE_BODY
            self.occupied_cells[(2, i)] = Cell.SNAKE_HEAD

        if self.with_boundaries:
            for i in range(self.matrix.shape[1]):
                self.matrix[0][i] = Cell.BLOCK_CELL
                self.matrix[self.matrix.shape[0]-1][i] = Cell.BLOCK_CELL
                self.occupied_cells[(0, i)] = Cell.BLOCK_CELL
                self.occupied_cells[(self.matrix.shape[0]-1, i)] = Cell.BLOCK_CELL

            for i in range(self.matrix.shape[0]):
                self.matrix[i][0] = Cell.BLOCK_CELL
                self.matrix[i][self.matrix.shape[1]-1] = Cell.BLOCK_CELL
                self.occupied_cells[(i, 0)] = Cell.BLOCK_CELL
                self.occupied_cells[(i, self.matrix.shape[1]-1)] = Cell.BLOCK_CELL

            self.occupied_cells[(0, None)] = None
            self.occupied_cells[(self.matrix.shape[0]-1, None)] = None
            self.occupied_cells[(None, 0)] = None
            self.occupied_cells[(None, self.matrix.shape[1]-1)] = None

        self.set_food()

    def set_food(self):
        while len(self.food) < self.num_agents:
            row = 0
            while (row, None) in self.occupied_cells:
                row = np.random.randint(1, self.height)
            column = 0
            while (row, column) in self.occupied_cells:
                column = np.random.randint(1, self.width)

            self.matrix[row, column] = Cell.FOOD
            self.occupied_cells[(row, column)] = Cell.FOOD
            self.food.append((row, column))

    def __str__(self):
        return_str = ""
        for row in self.matrix:
            for cell in row:
                return_str += str(Cell.CELL_REPRESENTATION[(cell[0], cell[1], cell[2])])
            return_str += '\n'
        return return_str

    def print(self):
        print(self)

    def step(self, actions):
        pass