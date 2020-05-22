from tetris import Tetris
import numpy as np

# This file is used for testing the environment and for training the internal networks

env = Tetris(train_reward = False, train_env = True, train_pregame = False, debug=True)

obs = env.reset()

ACTION_SPACE = env.action_space.n

action = np.random.randint(0, ACTION_SPACE - 1)

obs, reward, done, _ = env.step(action)

count = 0

while count != 3:
    while not done:
        action = np.random.randint(0, ACTION_SPACE - 1)
        obs, reward, done, _ = env.step(action)
        env.render()
    obs = env.reset()
    done = False
    count += 1