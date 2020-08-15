import keyboard as k
from Snake.environment import SnakeMaze
from Snake.variables import Direction
from Snake.env_renderer import CV2Renderer


env = SnakeMaze(50, 50, 1, with_boundaries=False)
env.reset()
renderer = CV2Renderer(env, delay=20)

while env.num_active_agents:
    renderer.render()
    snake_direction = env.snakes[0].direction
    if k.is_pressed("w"):
        snake_direction = Direction.NORTH
    elif k.is_pressed("d"):
        snake_direction = Direction.EAST
    elif k.is_pressed("s"):
        snake_direction = Direction.SOUTH
    elif k.is_pressed("a"):
        snake_direction = Direction.WEST
    env.step({0: snake_direction})
    renderer.render()