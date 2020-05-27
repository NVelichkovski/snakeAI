import os
from datetime import datetime

import numpy as np

from Snake.env_renderer import CV2Renderer
from Snake.environment import SnakeMaze
from utils import euclidean_distance, new_position
from Snake.variables import Status, Cell

env = SnakeMaze(20, 20, 1, with_boundaries=True)
env.reset()
renderer = CV2Renderer(env, image_size=(1000, 1000))

# while env.num_active_agents is not 0:
for _ in range(50):
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

timestamp = datetime.now().strftime('%d%h%Y__%H%M%S%f')
renderer.save_video(os.path.join('..', 'game_videos', 'euclidean_distance', timestamp + '.mp4'))
renderer.save_images(os.path.join('..', 'game_images', 'euclidean_distance', timestamp))
