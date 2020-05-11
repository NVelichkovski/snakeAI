import numpy as np

from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze
from utils import euclidean_distance, new_position
from Snake.variables import Status, Cell

env = SnakeMaze(30, 30, 2, with_boundaries=False)
env.reset()
renderer = CV2Renderer(env)

while env.num_active_agents is not 0:
    directions_dict = {}
    for handle, snake in env.snakes.items():

        if snake.status is Status.DEAD:
            continue

        possible_directions = [(snake.direction - 1) % 4, snake.direction, (snake.direction + 1) % 4]
        direction_dist = []

        for food_pos in env.food:
            snake_pos = snake.body[0]
            snake_direction = snake.direction

            new_positions = {direction: new_position(snake_pos, direction, env.matrix) for direction in
                             possible_directions}
            directions = []
            for direction in possible_directions:
                if (env.matrix[new_positions[direction]] == Cell.EMPTY_CELL).all() or (
                        env.matrix[new_positions[direction]] == Cell.FOOD).all():
                    directions.append((direction, euclidean_distance(new_positions[direction], food_pos)))

            direction_dist.append(
                min(directions, key=lambda x: x[1]) if len(directions) else (
                direction, euclidean_distance(new_positions[direction], food_pos)))

        best = np.argmin([t[1] for t in direction_dist])
        directions_dict[handle] = direction_dist[best][0]

    env.step(directions_dict)
    renderer.render()

renderer.destroy_window()
renderer.save_video('../game_videos')
renderer.save_images('../game_images')
