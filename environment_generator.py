import numpy as np

from agent import Snake

from typing import Dict, Tuple, Set

from variables import Cell, Actions


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
        self.num_active_agents = self.num_agents
        self.matrix = None
        self.with_boundaries = with_boundaries

        self.food = set()
        self.occupied_cells: Set[Tuple[int, int]] = {}

        self.snakes: Dict[int, Snake] = {i: Snake(i, self) for i in range(self.num_agents)}

    def reset(self):
        boundaries_offset = 2 if self.with_boundaries else 0
        self.matrix = np.zeros((self.width + boundaries_offset, self.height + boundaries_offset, 3), dtype=np.uint8)

        space_between_snakes = self.width // (self.num_agents + 2)

        num_snakes = None
        setting_snake_boundaries_offset = 1 if self.with_boundaries else 0
        for i in range(0 + setting_snake_boundaries_offset, self.width + setting_snake_boundaries_offset,
                       space_between_snakes):
            if num_snakes is None:
                # Skip the first snake
                num_snakes = 0
                continue

            if num_snakes == self.num_agents:
                # Skip the last snake
                break

            self.matrix[1][i] = Cell.SNAKE_BODY
            self.matrix[2][i] = Cell.SNAKE_HEAD
            self.occupied_cells[(1, i)] = Cell.SNAKE_BODY
            self.occupied_cells[(2, i)] = Cell.SNAKE_HEAD

            self.snakes[num_snakes].body = [(2, i), (1, i)]

            num_snakes += 1

        if self.with_boundaries:
            for i in range(self.matrix.shape[1]):
                self.matrix[0][i] = Cell.BLOCK_CELL
                self.matrix[self.matrix.shape[0] - 1][i] = Cell.BLOCK_CELL
                self.occupied_cells[(0, i)] = Cell.BLOCK_CELL
                self.occupied_cells[(self.matrix.shape[0] - 1, i)] = Cell.BLOCK_CELL

            for i in range(self.matrix.shape[0]):
                self.matrix[i][0] = Cell.BLOCK_CELL
                self.matrix[i][self.matrix.shape[1] - 1] = Cell.BLOCK_CELL
                self.occupied_cells[(i, 0)] = Cell.BLOCK_CELL
                self.occupied_cells[(i, self.matrix.shape[1] - 1)] = Cell.BLOCK_CELL

            self.occupied_cells[(0, None)] = None
            self.occupied_cells[(self.matrix.shape[0] - 1, None)] = None
            self.occupied_cells[(None, 0)] = None
            self.occupied_cells[(None, self.matrix.shape[1] - 1)] = None

        self.set_food()

    def set_food(self):
        while len(self.food) < self.num_active_agents:
            row = np.random.randint(1, self.height)
            while (row, None) in self.occupied_cells:
                row = np.random.randint(1, self.height)
            column = np.random.randint(1, self.height)
            while (row, column) in self.occupied_cells:
                column = np.random.randint(1, self.width)

            self.matrix[row, column] = Cell.FOOD
            self.occupied_cells[(row, column)] = Cell.FOOD
            self.food.add((row, column))

    def release_cells(self, cells_to_release: list):
        for cell in cells_to_release:
            self.occupied_cells[cell] = Cell.EMPTY_CELL

    def __str__(self):
        return_str = ""
        for row in self.matrix:
            for cell in row:
                return_str += str(Cell.CELL_REPRESENTATION[(cell[0], cell[1], cell[2])])
            return_str += '\n'
        return return_str

    def print(self):
        print(self)

    def step(self, actions: Dict[int, int] = {}):
        for handle, snake in self.snakes.items():
            action = actions[handle] if handle in actions else Actions.FORWARD
            snake.step(action)

        self.set_food()
