from tetris import Tetris
import numpy as np
import argparse

# This file is used for testing the environment and for training the internal networks

parser = argparse.ArgumentParser()
parser.add_argument('--reward', type=bool, default=False)
parser.add_argument('--env', type=bool, default=False)
parser.add_argument('--pregame', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

env = Tetris(train_reward = args.reward, train_env = args.env, train_pregame = args.pregame, debug = True, epochs=args.epochs)

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