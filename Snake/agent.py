from Snake.variables import Direction, Status, CellRenderEnc, Cell
from Snake.environment import *


class Snake:
    """
    Agent class.

    A snake object defines a single snake on the Snake. Each one is identified by a handle. The body is a list
    representing the cells that the snake occupies where the first cell is the snake's head. It can be faced in one
    of the following directions:
        - 0: North
        - 1: East
        - 2: South
        - 3: West

    And could be:
        - 0: Dead
        - 1: Active

    For now, the speed is constant.
    """

    def __init__(self, handle, env, direction=Direction.SOUTH, speed=1, body=[]):
        """
        Constructor for Snake object

        :param handle: int
        :param env: Environment
        :param direction: int, Optional
        :param speed: int
        :param body: Optional
        """
        self.handle = handle
        self.env: SnakeMaze = env

        self.direction = direction
        self.speed = speed
        self.body = body

        self.previous_direction = None
        self.length = 0
        self.status = Status.ACTIVE

    def kill_snake(self):
        """
        Release the cells occupied by the snake and update the snake's status.
        :return: None
        """
        self.status = Status.DEAD
        self.env.release_cells(self.body)
        self.env.num_active_agents -= 1
        self.body = None

    def step(self, direction):
        """
        Step the agent

        Make the provided action and update the matrix. If the maze is without boundaries the step handles transporting
        the snake from one side of the maze to the other.

        :param action: int
            The action that needs to be performed
        :return: list or None
            Returns the snake's body or None if the snake hits occupied cell
        """
        if self.status is Status.DEAD:
            return None

        if ((self.direction + 1) % 4) != direction and ((self.direction - 1) % 4) != direction:
            direction = self.direction
        head = self.body[0]
        offset_i, offset_j = Direction.COORDINATES_OFFSET[direction]
        head = ((head[0] + offset_i) % self.env.height, (head[1] + offset_j) % self.env.width)

        self.body.insert(0, head)
        released_cell = self.body.pop(-1)

        self.previous_direction = direction

        if self.env.matrix[head] == Cell.SNAKE_HEAD:
            self.body.append(released_cell)
            for snake in self.env.snakes.values():
                if snake.status is Status.DEAD:
                    continue
                if snake.body[0] == head:
                    snake.kill_snake()
        elif self.env.matrix[head] == Cell.FOOD:
            self.body.append(released_cell)
            self.env.food.remove(head)

            self.env.update_matrices(*self.body[0], 'SNAKE_HEAD')
            self.env.update_matrices(*self.body[1], 'SNAKE_BODY')
            self.env.snake_matrices[self.handle][self.body[0]] = CellRenderEnc.MY_SNAKE_HEAD
            self.env.snake_matrices[self.handle][self.body[1]] = CellRenderEnc.MY_SNAKE_BODY

        elif (self.env.matrix[head] == CellRenderEnc.EMPTY_CELL).all():
            self.env.update_matrices(*self.body[0], 'SNAKE_HEAD')
            self.env.update_matrices(*self.body[1], 'SNAKE_BODY')
            self.env.snake_matrices[self.handle][self.body[0]] = CellRenderEnc.MY_SNAKE_HEAD
            self.env.snake_matrices[self.handle][self.body[1]] = CellRenderEnc.MY_SNAKE_BODY

            self.env.release_cells([released_cell])

        else:
            self.body.append(released_cell)
            self.body.pop(0)
            self.kill_snake()

        self.direction = direction
        return self.body
