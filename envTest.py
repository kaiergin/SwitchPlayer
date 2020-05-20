from tetris import Tetris
import numpy as np

env = Tetris()

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