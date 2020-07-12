import numpy as np
from Snake.agent import Snake

from typing import Dict

from Snake.variables import CellRenderEnc, Cell


class SnakeMaze:
    """
    Snake Snake class.

    An Snake for multi-agent variation of the classic snake game. The goal of the game is to be the last living
    snake in the maze. Number of snakes as well as the dimensions of the maze is controlled with the following
    limitation:
        - Width: bigger than 5 since mazes smaller then 5x5 is not practical even for 1 agent
        - Height: bigger or equal then the width since there is no much difference between mazes of 25x50 or 50x25
        - The number of snakes: bigger than one but smaller then width/2. Initially, we need to have at least one cell
                                space between two snakes.

    Each time a snake eats a food cell, the snake gets bigger for one.

    """

    def __init__(self, width, height, max_num_agents, with_boundaries=True, cache_size=5):
        """
        SnakeMaze constructor

        :param width: int
            The width of the maze. Must be bigger then 5
        :param height: int
            The height of the maze. Bust be bigger or equal then the width
        :param max_num_agents: int
            Max number of agents. The actual number of the agents may be smaller.
        :param with_boundaries: boolean, Optional
            Default True. If with_boundaries is True there will be a wall around the maze.
        """
        if width < 5 or height < width:
            raise AttributeError(
                """
    Environment with his dimensions cannot be constructed.
    width > 5
    height >= width
    max_num_agents > 1
                """)
        self.with_boundaries = with_boundaries
        self.width = width
        self.height = height
        self.max_num_agents = max_num_agents
        self.num_agents = self.max_num_agents if self.max_num_agents * 2 < self.width + 2 else (self.width - 2) // 2
        self.num_active_agents = self.num_agents
        self.matrix = None
        self.snake_matrices = None

        self.food = set()

        self.snakes: Dict[int, Snake] = {i: Snake(i, self) for i in range(self.num_agents)}
        self.number_of_steps = 0

        self.image = None

        self.cache_size = cache_size
        self.cache = None

    def update_matrices(self, i: int, j: int, cell_type):
        self.matrix[i, j] = Cell.CELL_DICT[cell_type]
        for snake in self.snake_matrices:
            snake[i, j] = CellRenderEnc.CELL_DICT[cell_type]

    def reset(self):
        """
        Reset the Snake.

        A matrix is constructed by the width and height specified from the attributed in the constructor. Then all
        snakes are equally distributed on the top of the maze facing south. Finally, food is placed on random positions
        around the maze.

        :return: None
        """
        boundaries_offset = 2 if self.with_boundaries else 0
        self.matrix = np.zeros((self.width + boundaries_offset, self.height + boundaries_offset), dtype='float32')
        self.snake_matrices = np.zeros(
            (self.num_agents + 1, self.width + boundaries_offset, self.height + boundaries_offset, 3), dtype='float32')
        self.cache = [np.zeros(
            (self.num_agents + 1, self.width + boundaries_offset, self.height + boundaries_offset, 3), dtype='float32') for _ in range(self.cache_size)]

        space_between_snakes = self.width // (self.num_agents + 2)

        num_snakes = None
        setting_snake_boundaries_offset = 1 if self.with_boundaries else 0

        handle = 0
        for i in range(0 + setting_snake_boundaries_offset, self.width + setting_snake_boundaries_offset,
                       space_between_snakes):
            if num_snakes is None:
                # Skip the first snake
                num_snakes = 0
                continue

            if num_snakes == self.num_agents:
                # Skip the last snake
                break

            self.update_matrices(1, i, 'SNAKE_HEAD')
            self.update_matrices(2, i, 'SNAKE_BODY')

            self.snake_matrices[handle][1, i] = CellRenderEnc.MY_SNAKE_HEAD
            self.snake_matrices[handle][2, i] = CellRenderEnc.MY_SNAKE_BODY
            handle += 1

            self.snakes[num_snakes].body = [(2, i), (1, i)]
            num_snakes += 1

        if self.with_boundaries:
            for i in range(self.matrix.shape[1]):
                self.update_matrices(0, i, 'BLOCK_CELL')
                self.update_matrices(self.matrix.shape[0] - 1, i, 'BLOCK_CELL')

            for i in range(self.matrix.shape[0]):
                self.update_matrices(i, 0, 'BLOCK_CELL')
                self.update_matrices(i, self.matrix.shape[1] - 1, 'BLOCK_CELL')
        self.set_food()

    def set_food(self):
        """
        Set food for every snake randomly around the matrix.

        The number of food cells could be bigger than the number of snakes when a snake die.
        :return: None
        """
        while len(self.food) < self.num_active_agents:
            row = np.random.randint(1, self.height)
            column = np.random.randint(1, self.height)
            while not (self.matrix[row, column] == Cell.EMPTY_CELL).all():
                row = np.random.randint(1, self.height)
                column = np.random.randint(1, self.height)

            self.update_matrices(row, column, 'FOOD')
            self.food.add((row, column))

    def release_cells(self, cells_to_release: list):
        """
        Set the provided cells as empty

        :param cells_to_release: list
            List of cells to be released
        :return:
        """
        for cell in cells_to_release:
            if (self.matrix[cell] == Cell.BLOCK_CELL).all():
                continue
            self.update_matrices(*cell, 'EMPTY_CELL')

    def __str__(self):
        return_str = ""
        for row in self.matrix:
            for cell in row:
                return_str += str(Cell.CELL_REPRESENTATION[cell])
            return_str += '\n'
        return return_str

    def step(self, directions: Dict[int, int] = {}):
        """Step the Environment.

        Call the Agent.step for each snake with the new direction provided in the directions param.
        If there is no action provided the snake continue Forward.
        :param directions: Dict[int, int], Optional
        :return: None
        """
        if self.num_active_agents is 0:
            return

        for handle, snake in self.snakes.items():
            direction = directions[handle] if handle in directions else snake.direction
            snake.step(direction)

        self.set_food()
        self.number_of_steps += 1

        if len(self.cache) >= self.cache_size:
            self.cache.pop(0)
        self.cache.append(self.snake_matrices.copy())
