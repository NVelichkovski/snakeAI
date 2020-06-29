import os

import numpy as np

from Snake.environment import SnakeMaze
from Snake.utils import resize_image, save_images, save_video, save_graph, save_eval
from Model.replay_memory import ReplayMemory, Experience

import tensorflow as tf


@tf.function
def loss(p, t):
    return tf.reduce_sum(tf.square(t - p))


@tf.function
def train_step(states, targets, model, optimizer):
    with tf.GradientTape() as tape:
        q = model(states, training=True)
        _loss = loss(q, targets)
    grad = tape.gradient(_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))


def train_dqn(model, optimizer, reward, **kwargs):
    """
            Training model

            :param model: tf.keras.Model
                An instance ovr the keras.Model
            :param reward: function
                A function that returns the reward of the current state

                Args:
                    ::param snake: Snake
                    ::param env: SnakeMaze
                    ::param direction: Direction

                Returns:
                    int: The current reward

            :param optimizer: tf.keras.Optimizer
                The optimizer used to train the model

            :param kwargs:
                save_images (bool): False
                    If true images are saved on the eval episode

                save_videos (bool): False
                    If true videos are saved on the eval episode

                save_models (bool): False
                    If true Model are saved on the eval episode

                save_graphs (bool): False
                    If true graphs are saved on the eval episode

                evaluate_each (int): 50
                    Number of episodes between two evaluations

                num_rolling_avg_sample (int): 50
                    Number of episodes that are used for the rolling average

                max_steps_per_episode (int): 200
                    Max number of steps in each episode

                num_episodes (int, None): None
                    Number of episodes to train, if None runs forever

                gamma (float): .8
                    The discount factor

                epsilon (float): 1.
                    The exploration/exploitation rate

                epsilon_decay (float): 0.0005
                    The decay of the exploration/exploitation rate

                min_epsilon (float): 0.01
                    The epsilon convergence

                memory_size (int): 10000
                    Capacity of the reply memory, if less then 50 Replay Memory will not be used

                boundaries (bool): True
                    Include boundaries in the maze

                maze_width (int): 10
                    The width of the maze

                maze_height (int, None): None
                    The height of the maze, if not specified the maze is squared

                max_snakes (int): 1
                    The max number of snakes in the environment

                path_to_weights (str): None
                    Path to saved weights for the model

                image_size (tuple[int, int]): (112, 112)
                    The size of the image used as input in the model

                verbose (bool): True
                    Print info about the training

                training_dir (str): './'
                    Path to the directory for training
            """
    verbose = kwargs['verbose'] if 'verbose' in kwargs else True
    save_videos_ = kwargs['save_videos'] if 'save_videos' in kwargs else False
    save_images_ = kwargs['save_images'] if 'save_images' in kwargs else False
    evaluate_each = kwargs['evaluate_each'] if 'evaluate_each' in kwargs else 50
    num_rolling_avg_sample = kwargs['num_rolling_avg_sample'] if 'num_rolling_avg_sample' in kwargs else 50
    max_steps_per_episode = kwargs['max_steps_per_episode'] if 'max_steps_per_episode' in kwargs else 200
    num_episodes = kwargs['num_episodes'] if 'num_episodes' in kwargs else None
    gamma = kwargs['gamma'] if 'gamma' in kwargs else .8
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1.
    epsilon_decay = kwargs['epsilon_decay'] if 'epsilon_decay' in kwargs else 5e-4
    min_epsilon = kwargs['min_epsilon'] if 'min_epsilon' in kwargs else 1e-2
    memory_size = kwargs['memory_size'] if 'memory_size' in kwargs else 10000
    boundaries = kwargs['boundaries'] if 'boundaries' in kwargs else True
    maze_width = kwargs['maze_width'] if 'maze_width' in kwargs else 10
    maze_height = kwargs['maze_height'] if 'maze_height' in kwargs else maze_width
    max_snakes = kwargs['max_snakes'] if 'max_snakes' in kwargs else 1
    path_to_weights = kwargs['path_to_weights'] if 'path_to_weights' in kwargs else None
    image_size = kwargs['image_size'] if 'image_size' in kwargs else (112, 112)
    training_dir = kwargs['training_dir'] if 'training_dir' in kwargs else './'
    comment = kwargs['comment'] if 'comment' in kwargs else ''
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    
    comment = comment + f"""

    _________________________________________________________
    ---------------------------------------------------------

    max_steps_per_episode  --> {max_steps_per_episode}
    num_episodes           --> {num_episodes}
    gamma                  --> {gamma}
    epsilon                --> {epsilon}
    epsilon_decay          --> {epsilon_decay}
    min_epsilon            --> {min_epsilon}
    memory_size            --> {memory_size}
    boundaries             --> {boundaries}
    maze_width             --> {maze_width}
    maze_height            --> {maze_height}
    max_snakes             --> {max_snakes}
    path_to_weights        --> {path_to_weights}
    image_size             --> {image_size}
    _________________________________________________________
    ---------------------------------------------------------
    """

    os.makedirs(training_dir, exist_ok=True)
    comment_path = os.path.join(training_dir, "info.txt")
    with open(comment_path, "w") as text_file:
        text_file.write(comment)
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

    if path_to_weights is not None:
        model.load_weights(path_to_weights)

    episode_number = 0

    rolling_avg_reward = np.array([])
    rolling_avg_epsilon = np.array([])


    reward_window = np.array([])

    memory = ReplayMemory(capacity=memory_size) if (memory_size > 50 and memory_size > batch_size) else None

    while num_episodes is None or episode_number < num_episodes:
        if verbose:
            print("____________________________________________________________________________________________")
        is_eval_episode = (episode_number % evaluate_each == 0) or (num_episodes and episode_number == num_episodes - 1)
        episode_images = []

        env = SnakeMaze(
            width=maze_width,
            height=(maze_height if maze_height is not None else maze_width),
            max_num_agents=max_snakes,
            with_boundaries=boundaries)
        env.reset()
        episode_reward = 0

        for _ in range(max_steps_per_episode):

            if env.num_active_agents == 0:
                break

            state = resize_image(env.snake_matrices[0], image_size)

            if is_eval_episode and (save_images_ or save_videos_):
                episode_images.append(state)

            q = model(state.reshape((-1, *state.shape)))
            direction = np.random.randint(4) if np.random.rand(1) < epsilon else np.argmax(q)
            env.step({0: direction})
            current_reward = reward(env.snakes[0], env, direction)
            state2 = resize_image(env.snake_matrices[0], image_size)

            episode_reward += current_reward

            if reward_window is None:
                reward_window = np.array([current_reward])
            else:
                reward_window = np.append(reward_window, current_reward)

            if memory is not None:
                memory.push(Experience(state, direction, state2, current_reward))

                if np.random.rand() > memory.space():
                    continue
                batches_info = memory.pop()
            else:
                batches_info = [Experience(state, direction, state2, current_reward)]

            states, q_targets = [], []
            for state, direction, state2, current_reward in batches_info:
                q_target = model(state2.reshape((-1, *state.shape)))
                max_q = np.max(np.max(q_target))
                q_target = q_target.numpy()
                q_target[0, direction] = current_reward + gamma * max_q

                states.append(state)
                q_targets.append(q_target)
            train_step(np.array(states), np.array(q_targets), model, optimizer)

        if verbose:
            print(f"Episode {episode_number + 1} Done!")
            print(f"Episode reward: {episode_reward}")
            print(f"Epsilon: {epsilon}")
            if memory is not None:
                print(f"Replay Memory size: {len(memory.experiences)}")

        if len(reward_window) >= num_rolling_avg_sample:
            rolling_avg = np.mean(reward_window)
            reward_window = np.delete(reward_window, 0)
            rolling_avg_reward = np.append(rolling_avg_reward, rolling_avg)
            rolling_avg_epsilon = np.append(rolling_avg_epsilon, epsilon)


        epsilon = max(epsilon - epsilon_decay, min_epsilon)

        if verbose:
            print()

        if is_eval_episode:
            save_eval(model, episode_number, episode_images, rolling_avg_reward, rolling_avg_epsilon, **kwargs)

        episode_number += 1

    return model
