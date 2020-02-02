from variables import Direction, Status, Actions, Cell
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

    def kill_snake(self):
        self.status = Status.DEAD
        self.env.release_cells(self.body)
        self.env.num_active_agents -= 1
        self.body = None

    def step(self, action):
        """
            return None if snake hit occupied cell else returns snake body
        """
        if self.status is Status.DEAD:
            return None
        head = self.body[0]
        offset_i, offset_j = Actions.COORDINATES_OFFSET[self.direction][action]
        head = ((head[0] + offset_i) % self.env.height, (head[1] + offset_j) % self.env.width)

        self.body.insert(0, head)
        released_cell = self.body.pop(-1)

        self.previous_action = action

        if (self.env.matrix[head] == Cell.SNAKE_HEAD).all():
            self.body.append(released_cell)
            for snake in self.env.snakes.values():
                if snake.body[0] == head:
                    snake.kill_snake()
        elif (self.env.matrix[head] == Cell.FOOD).all():
            self.body.append(released_cell)
            self.env.food.remove(head)
            self.env.matrix[self.body[0]] = Cell.SNAKE_HEAD
            self.env.matrix[self.body[1]] = Cell.SNAKE_BODY

        elif (self.env.matrix[head] == Cell.EMPTY_CELL).all():
            self.env.matrix[self.body[0]] = Cell.SNAKE_HEAD
            self.env.matrix[self.body[1]] = Cell.SNAKE_BODY
            self.env.release_cells([released_cell])

        else:
            self.body.append(released_cell)
            self.body.pop(0)
            self.kill_snake()
        self.direction = (self.direction + action) % 4
        return self.body
