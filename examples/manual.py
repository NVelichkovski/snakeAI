from Snake.environment import SnakeMaze
from Snake.env_renderer import CV2Renderer
from Snake.variables import Direction

import keyboard as k

env = SnakeMaze(30, 30, 1)
env.reset()

renderer = CV2Renderer(env)

while env.num_active_agents:
    renderer.render()

    direction = env.snakes[0].direction
    if k.is_pressed('w'):
        direction = Direction.NORTH
    elif k.is_pressed('d'):
        direction = Direction.EAST
    elif k.is_pressed('s'):
        direction = Direction.SOUTH
    elif k.is_pressed('a'):
        direction = Direction.WEST

    env.step({0: direction})
    renderer.render()
