import random
import os
import shutil
from PIL import Image
#from tetris import Tetris
import gym
import numpy as np
import pickle

# This file saves entire games played in order
# It also loads replays of games for easy access

# To do:
# - add a cache for faster access time
# - add prioritized storage
# - Parallelize image loading for batch training

storage_path = 'game_storage/'
max_games = 50

class Storage:
    def __init__(self, max_g = max_games, debug = False):
        self.max_games = max_g
        self.debug = debug
        try:
            self.game_list = pickle.load(open('game_list.p', 'rb'))
            self.current_folder = self.game_list[-1]
            self.current_index = 0
            self.reward_file = open(storage_path + str(self.current_folder) + '/reward.csv', 'w')
            self.action_file = open(storage_path + str(self.current_folder) + '/action.csv', 'w')
        except:
            self.game_list = []
            self.reward_file = None
            self.action_file = None
            self.new_game()
    
    # Takes in a numpy array, converts to PIL image, and saves it to the current game directory with the current index
    # Appends reward for frame onto reward file
    # (optional) if action added, appends to action file
    def save_frame(self, frame, reward, action=-1):
        im = Image.fromarray(frame)
        im = im.resize((95,95))
        im.save(storage_path + str(self.current_folder) + '/' + str(self.current_index) + '.png')
        self.reward_file.write(str(int(round(reward))) + ', ')
        if action != -1:
            self.action_file.write(str(action) + ', ')
        self.current_index += 1
    
    # Creates new folder and resets index
    def new_game(self):
        try:
            self.reward_file.close()
            self.action_file.close()
        except:
            pass
        self.current_folder = random.randrange(2147483647)
        self.game_list.append(self.current_folder)
        self.current_index = 0
        os.mkdir(storage_path + str(self.current_folder))
        self.reward_file = open(storage_path + str(self.current_folder) + '/reward.csv', 'w')
        self.action_file = open(storage_path + str(self.current_folder) + '/action.csv', 'w')
        self.free_space()
        pickle.dump(self.game_list, open('game_list.p', 'wb'))
        self.current_frames = []
        self.current_rewards = []
        self.current_actions = []
    
    def get_current_folder(self):
        return str(self.current_folder)

    # game: str
    def get_game_length(self, game):
        if os.path.exists(storage_path + game):
            return len(os.listdir(storage_path + game)) - 2
        else:
            return -1

    # Loads BUFFER_SIZE amount of frames from GAME, where the last frame in the buffer is INDEX
    # buffer_size: int,  game: str,  index: int, future_steps: int
    def load_replay(self, buffer_size, game, index, future_steps=0):
        game_len = self.get_game_length(game)
        if game_len - 1 <= index:
            return None
        else:
            with open(storage_path + game + '/reward.csv', 'r') as f:
                all_rewards = f.read().split(',')
            with open(storage_path + game + '/action.csv', 'r') as f:
                all_actions = f.read().split(',')
            buffer = []
            rewards = []
            actions = []
            for x in range((index - buffer_size) + 1, index + 1):
                if x < 1:
                    y = 1
                else:
                    y = x
                buffer.append(np.asarray(Image.open(storage_path + game + '/' + str(y) + '.png'), dtype=np.float32))
                rewards.append(float(all_rewards[y].strip()))
                actions.append(int(all_actions[y-1].strip()))
            future_frames = []
            future_rewards = []
            future_actions = []
            not_done = 1.0
            for x in range(index + 1, index + 1 + future_steps):
                # If this fails, means game is done because there are no more frames/rewards/actions with that name
                if x >= game_len:
                    not_done = 0.0
                    break
                future_frames.append(np.asarray(Image.open(storage_path + game + '/' + str(x) + '.png'), dtype=np.float32))
                future_rewards.append(float(all_rewards[x].strip()))
                future_actions.append(int(all_actions[x - 1].strip()))
            return np.stack(buffer, 0), np.array(rewards), np.array(actions), \
                        (np.stack(future_frames, 0), np.array(future_rewards, dtype=np.float32), np.array(future_actions), np.array(not_done, dtype=np.float32)), (game, index)
    
    # Loads BUFFER_SIZE amount of frames (in order) from a random game
    # buffer_size: int
    def load_random_replay(self, buffer_size, future_steps=1):
        game_length = 0
        game = ''
        # Buffer size + 1 because we need to load at least buffer size frames + 1 action taken
        while game_length < buffer_size + 1 or game == str(self.current_folder):
            game = random.choice(os.listdir(storage_path)).split(os.pathsep)[-1]
            game_length = self.get_game_length(game)
        index = random.randrange(buffer_size, game_length-4)
        return self.load_replay(buffer_size, game, index, future_steps=future_steps)

    def free_space(self):
        if len(self.game_list) > self.max_games:
            # Remove oldest game
            try:
                shutil.rmtree(storage_path + str(self.game_list[0]))
            except:
                print('Error popping folder:', str(self.game_list[0]))
            self.game_list.pop(0)

# Testing
if __name__ == '__main__':
    storage = Storage(max_g = 3, debug = True)
    env = gym.make('MsPacman-v0')
    for x in range(5):
        obs = env.reset()
        action = random.randint(0,env.action_space.n - 1)
        reward = 0.0
        storage.save_frame(obs, reward, action)
        done = False
        while not done:
            obs, reward, done, _ = env.step(action)
            action = random.randint(0,env.action_space.n - 1)
            storage.save_frame(obs, reward, action)
        previous_game = storage.get_current_folder()
        storage.new_game()
    for x in range(10):
        try:
            frames, rewards, actions, future = storage.load_replay(16, previous_game, storage.get_game_length(previous_game)-x, future_steps=3)
            print('-- Current replay index:', storage.get_game_length(previous_game)-x, '--')
            print('Future lengths:')
            for x in future:
                try:
                    print(len(x))
                except:
                    print(x)
            print('Frame shape:', frames.shape)
            print('reward shape:', rewards.shape)
            print('action shape:', actions.shape)
        except:
            print('Could not load replay')
    for x in range(10):
        frames, rewards, actions, future = storage.load_replay(16, previous_game, x, future_steps=5)
        print('-- Current replay index:', storage.get_game_length(previous_game)-x, '--')
        print('Future shapes:')
        for x in future:
            try:
                print(len(x))
            except:
                print(x)
        print('Frame shape:', frames.shape)
        print('reward shape:', rewards.shape)
        print('action shape:', actions.shape)