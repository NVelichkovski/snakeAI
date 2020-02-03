import numpy as np

from environment.env_renderer import EnvRenderer
from environment.environment_generator import Environment
from utils import euclidean_distance, new_position
from environment.variables import Status, Cell

for _ in range(1):
    env = Environment(30, 30, 2)
    env.reset()
    renderer = EnvRenderer(env)

    while env.num_active_agents is not 0:
        actions_dict = {}
        for handle, snake in env.snakes.items():
            if snake.status is Status.DEAD:
                continue
            action_dist = []
            for food_pos in env.food:
                snake_pos = snake.body[0]
                snake_direction = snake.direction

                new_positions = {action: new_position(snake_pos, snake_direction, action, env.matrix) for action in
                                 range(-1, 2)}
                actions = []
                for action in range(-1, 2):
                    if (env.matrix[new_positions[action]] == Cell.EMPTY_CELL).all() or (
                            env.matrix[new_positions[action]] == Cell.FOOD).all():
                        actions.append(euclidean_distance(new_positions[action], food_pos))
                    else:
                        actions.append(env.matrix.shape[0] * env.matrix.shape[1])
                action = np.argmin(actions) - 1
                action_dist.append((action, min(actions)))

            best = np.argmin([t[1] for t in action_dist])
            actions_dict[handle] = action_dist[best][0]

        env.step(actions_dict)
        renderer.render()

    renderer.destroy_window()
    renderer.save_video('../game_videos')
    renderer.save_images('../game_images')
