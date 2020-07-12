from Snake.environment import SnakeMaze
from Snake.env_renderer import CV2Renderer
from Snake.utils import euclidean_distance

from Snake.variables import Direction, Cell

env = SnakeMaze(40, 40, 1)
env.reset()

renderer = CV2Renderer(env)
while env.num_active_agents != 0:

    snake_position = env.snakes[0].body[0]
    snake_direction = env.snakes[0].direction
    food_position = env.food[0]

    actions = []

    for direction_change in [-1, 0, 1]:

        new_direction = (snake_direction + direction_change) % 4

        direction_offset = Direction.COORDINATES_OFFSET[new_direction]
        new_position = snake_position[0] + direction_offset[0], snake_position[1] + direction_offset[1]

        if env.matrix[new_position] not in [Cell.FOOD, Cell.EMPTY_CELL]:
            continue

        actions.append((new_direction, euclidean_distance(food_position, new_position)))
    best_action = min(actions, key=lambda x: x[1])[0] if len(actions) else snake_direction
    env.step({0: best_action})
    renderer.render()

