from __future__ import division
import argparse
import sys
sys.path.append('keras-rl')
from PIL import Image
import numpy as np
import tetris
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input
from keras.optimizers import Adam
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import PrioritizedMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.layers import NoisyNetDense
import os
import glob

#We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
#a sense of direction and speed.
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

#Standard Atari processing
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='Tetris99')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = tetris.Tetris()
np.random.seed(231)
env.seed(231)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

#Standard DQN model architecture, but swapping the Dense classifier layers for the rl.layers.NoisyNetDense version.
input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame = Input(shape=(input_shape))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense= Flatten()(cv3)
dense = NoisyNetDense(512, activation='relu')(dense)
buttons = NoisyNetDense(nb_actions, activation='linear')(dense)
model = Model(inputs=frame,outputs=buttons)
print(model.summary())

memory = PrioritizedMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=30000000, window_length=WINDOW_LENGTH)

processor = AtariProcessor()

#This is the important difference. Rather than using an E Greedy approach, where
#we keep the network consistent but randomize the way we interpret its predictions,
#in NoisyNet we are adding noise to the network and simply choosing the best value.
policy = GreedyQPolicy()

#N-step loss with n of 3
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=10000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., n_step=3, custom_model_objects={"NoisyNetDense":NoisyNetDense})

#Prioritized Memories typically use lower learning rates
dqn.compile(Adam(lr=.00025/4), metrics=['mae'])

folder_path = 'model_saves/'

if args.mode == 'train':
    checkpoint_weights_filename = folder_path + 'advanced_dqn_' + args.env_name + '_weights_{step}.h5f'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
    list_of_files = glob.glob(folder_path + '*')
    if len(list_of_files) != 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        dqn.load_weights(latest_file)
        print('-- Loaded DQN weights :', latest_file, '--')
    dqn.fit(env, callbacks=callbacks, nb_steps=30000000, verbose=1)#, nb_max_episode_steps=20000)


elif args.mode == 'test':
    # 30 million steps
    weights_filename = folder_path + 'advanced_dqn_' + args.env_name + '_weights_30000000.h5f'
    if args.weights:
        # Choose a step
        weights_filename = weights_filename = folder_path + 'advanced_dqn_' + args.env_name + '_weights_' + args.weights + '.h5f'
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=20, visualize=True, nb_max_start_steps=80)
