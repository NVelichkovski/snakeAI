from constants import Direction, Status, Actions, Cell
from environment_generator import *

class Snake:
    def __init__(self, handle, env, direction=Direction.SOUTH, speed=1, body=[]):
        self.handle = handle
        self.env: Environment = env

        self.direction = direction
        self.speed = speed
        self.body = body

        self.previous_action = None
        self.length = 0
        self.status = Status.ACTIVE

    def kill_snake(self, released_cell=[]):
        self.status = Status.DEAD
        self.body.extend(released_cell)
        self.env.release_cells(self.body)
        self.env.num_active_agents -= 1

        return None

    def step(self, action):
        """
            return None if snake hit occupied cell else returns snake body
        """

        head = self.body[0]
        self.env.matrix[head] = Cell.SNAKE_BODY

        offset_i, offset_j = Actions.COORDINATES_OFFSET[self.direction][action]
        head = (head[0] + offset_i, head[1] + offset_j)

        self.body.insert(0, head)
        self.env.matrix[head] = Cell.SNAKE_HEAD

        released_cell = self.body.pop(-1)
        self.env.matrix[released_cell] = Cell.EMPTY_CELL

        self.previous_action = action

        if head in self.env.occupied_cells:
            if self.env.occupied_cells[head] is not Cell.FOOD:

                if self.env.occupied_cells[head] is Cell.SNAKE_HEAD:
                    # If 2 or more snakes end up with their heads on the same cell in the same time kill all if them
                    for snake in self.env.snakes.values():
                        if snake.body[0] == head:
                            snake.kill_snake()

                return self.kill_snake([released_cell])

            elif self.env.occupied_cells[head] is Cell.FOOD:
                self.body.append(released_cell)
                self.env.food.remove(head)

        return self.body
