from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Convolution2D, Input, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
from core import Env
from controller import sync, send_cmd, tetris_enum, p_wait, BTN_A, BTN_B, BTN_R, BTN_L, DPAD_U
import cv2
from gym import spaces
import pathlib
from statistics import mean
from random import shuffle, choice, randrange

RENDER = True
# Saves all images that are evaluated as line clear or tetris
# Helps me continuously improve the reward function
SAVE_REWARD = True

TRAINING_BUFFER = 20
INPUT_SHAPE = (90,160)
OUTPUT_ENV = 5
OUTPUT_REWARD = 3
OUTPUT_PRE = 2
ACTION_SPACE = len(tetris_enum)

CLASS_NAMES_ENV = np.array(['neutral', 'end_screen', 'error', 'title_a', 'title_b'])
CLASS_NAMES_PRE = np.array(['go', 'pregame'])
CLASS_NAMES_REWARD = np.array(['neutral', 'double', 'tetris'])

# Data paths
env_data = pathlib.Path('training/tetris_environment')
reward_data = pathlib.Path('training/tetris_line_clears')
pregame_data = pathlib.Path('training/tetris_pregame_opt')

# Model save paths
env_path = 'tetris/environment/env.ckpt'
reward_path = 'tetris/reward/reward.ckpt'
pregame_path = 'tetris/pregame/pregame.ckpt'

# Helper functions

def swap(a,b): return b, a

def one_hot(a, num_classes): return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def preprocess(im): return np.expand_dims(im.astype('float32') / 255.0, axis = 0)

# This class is a wrapper for an OpenAI gym environment
class Tetris(Env):
    # -1 so that the bot cannot choose BTN_NONE
    action_space = spaces.Discrete(ACTION_SPACE - 1)

    def __init__(self, train_reward=False, train_env=False, train_pregame=False, capture_device=0, epochs=20, debug=False):
        self.DEBUG = debug
        self.clock = 0
        self.cumulative_reward = 0
        self.reward = self.model_reward()
        self.env = self.model_environment()
        self.pregame = self.model_pregame()
        self.rewarded_steps = 0
        # Sync to the switch so that we can send controller commands
        if self.DEBUG:
            print('Attempting to sync to MCU')
        sync()
        if self.DEBUG:
            print('Successfully synced to MCU')
        # Connect controller
        send_cmd(BTN_L + BTN_R)
        p_wait(0.5)
        send_cmd()
        p_wait(0.1)
        send_cmd(BTN_A)
        p_wait(0.5)
        send_cmd()
        if self.DEBUG:
            print('Successfully paired controller')
        # Capture switch screen
        try:
            self.switch_screen = cv2.VideoCapture(capture_device)
        except:
            print('Unable to open capture device. Is it plugged in?')
            exit()
        if train_env:
            print('Starting environment training')
            self.fit_environment(epochs)
            print('Finished environment training - saving')
            print('You can now set train_environment = False')
            self.env.save_weights(env_path)
        if train_pregame:
            print('Starting pregame optimization training')
            self.fit_pregame(epochs)
            print('Finished optimization training - saving')
            print('You can now set train_pregame = False')
            self.pregame.save_weights(pregame_path)
        if train_reward:
            print('Starting reward training')
            self.fit_reward(epochs)
            print('Finished reward training - saving')
            print('You can now set train_reward = False')
            self.reward.save_weights(reward_path)
        
    # A model that determines what state the game is in
    # (IN_GAME, RESULT_SCREEN, ERROR_SCREEN, TITLE_A, TITLE_B)
    def model_environment(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=2, activation='relu')(frame)
        x = MaxPooling2D((5,5))(x)
        x = Convolution2D(32, kernel_size=(4,4), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_ENV, activation='softmax')(x)
        m = Model(inputs = frame, outputs = output)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        try:
            m.load_weights(env_path)
        except:
            print('Unable to load environment network')
        return m

    # A model for determining reward
    # (neutral [no reward], line cleared [1 point], tetris [1 point])
    def model_reward(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=2, activation='relu')(frame)
        x = MaxPooling2D((5,5))(x)
        x = Convolution2D(32, kernel_size=(4,4), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_REWARD, activation='softmax')(x)
        m = Model(inputs = frame, outputs = output)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if self.DEBUG:
            print('REWARD SUMMARY')
            m.summary()
        try:
            m.load_weights(reward_path)
        except:
            print('Unable to load reward network')
        return m

    # A model for waiting until the GO screen
    # (GO screen, pregame waiting screen)
    def model_pregame(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=2, activation='relu')(frame)
        x = MaxPooling2D((5,5))(x)
        x = Convolution2D(32, kernel_size=(4,4), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_PRE, activation='softmax')(x)
        m = Model(inputs = frame, outputs = output)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        try:
            m.load_weights(pregame_path)
        except:
            print('Unable to load pregame-optimization network')
        return m

    def fit_reward(self, epochs = 3):
        for i in range(epochs):
            batch = np.zeros((10, *INPUT_SHAPE, 3))
            labels = np.zeros((10, OUTPUT_REWARD))
            x = 0
            paths = list(reward_data.glob('*/*.png'))
            shuffle(paths)
            accuracy = 0
            total = 0
            for path in paths:
                path = str(path)
                # Already resized
                im = cv2.imread(path)
                if im is None:
                    continue
                if im.size == 0:
                    continue
                batch[x] = im / 255.0
                # Split by windows path divider
                label = path.split('\\')[2]
                found = np.where(CLASS_NAMES_REWARD == label)
                labels[x] = one_hot(found[0][0], CLASS_NAMES_REWARD.size)
                x += 1
                if x == 10:
                    accuracy += self.reward.train_on_batch(x = batch, y = labels)[1]
                    total += 1
                    x = 0
            if i == epochs - 1:
                print('Final accuracy reward:', accuracy/total)
    
    def fit_environment(self, epochs = 3):
        for _ in range(epochs):
            batch = np.zeros((10, *INPUT_SHAPE, 3))
            labels = np.zeros((10, OUTPUT_ENV))
            x = 0
            paths = list(env_data.glob('*/*.png'))
            shuffle(paths)
            for path in paths:
                path = str(path)
                # Already resized
                try:
                    im = cv2.imread(path)
                except:
                    print('Unable to read in image')
                    continue
                batch[x] = im / 255.0
                # Split by windows path divider
                label = path.split('\\')[2]
                found = np.where(CLASS_NAMES_ENV == label)
                labels[x] = one_hot(found[0][0], CLASS_NAMES_ENV.size)
                x += 1
                if x == 10:
                    self.env.train_on_batch(x = batch, y = labels)
                    x = 0

    def fit_pregame(self, epochs = 3):
        for _ in range(epochs):
            batch = np.zeros((10, *INPUT_SHAPE, 3))
            labels = np.zeros((10, OUTPUT_PRE))
            x = 0
            paths = list(pregame_data.glob('*/*.png'))
            shuffle(paths)
            for path in paths:
                path = str(path)
                # Already resized
                try:
                    im = cv2.imread(path)
                except:
                    print('Unable to read in image')
                    continue
                batch[x] = im / 255.0
                # Split by windows path divider
                label = path.split('\\')[2]
                found = np.where(CLASS_NAMES_PRE == label)
                labels[x] = one_hot(found[0][0], CLASS_NAMES_PRE.size)
                x += 1
                if x == 10:
                    self.pregame.train_on_batch(x = batch, y = labels)
                    x = 0

    def step(self, action):
        # Send the chosen button press to switch
        send_cmd(tetris_enum[action])
        p_wait(.05)
        if tetris_enum[action] == DPAD_U:
            self.rewarded_steps = 7
        # Read in the resulting frame
        ret_val, self.current_frame = self.switch_screen.read()
        if RENDER:
            self.render()
        if ret_val == False:
            print('Error reading from switch screen. Switch disconnected?')
            print('Exiting')
            exit()
        
        img_resize = cv2.resize(self.current_frame, swap(*INPUT_SHAPE))
        im = Image.fromarray(img_resize)
        # Necessary because im.save saves as BGR instead of RGB
        im.save('temp.png')
        # im_import is different than im (which is annoying but it is what it is)
        im_import = cv2.imread('temp.png')

        preprocessed = preprocess(im_import)

        # Evaluate on environment network
        env_out = np.argmax(self.env.predict_on_batch(preprocessed))
        if self.DEBUG and self.clock % 10 == 0:
            print('Current environment evaluation:', CLASS_NAMES_ENV[env_out])
        done = False
        if env_out == 1:
            # Endgame screen
            reward = -1.0
            done = True
        elif env_out != 0:
            # Not in-game
            reward = 0.0
            done = True
        else:
            reward = 0.0
            if self.rewarded_steps != 0:
                reward_func = np.argmax(self.reward.predict_on_batch(preprocessed))
                if reward_func != 0: # not neutral
                    reward = 1.0
                self.rewarded_steps -= 1
                if self.DEBUG:
                    print('Current reward:', reward)
        self.clock += 1
        # Crop to get only area of tetris
        cropped = self.current_frame[:, 480:-480]
        #cv2.imshow('cropped', cropped)
        #cv2.waitKey(1)
        if reward == 1.0 and SAVE_REWARD:
            im.save('training/bot_data/' + str(randrange(2147483647)) + '.png')
        self.cumulative_reward += reward
        return cropped, reward, done, {}

    def reset(self):
        self.clock = 0
        if self.DEBUG:
            print('Previous cumulative reward:', self.cumulative_reward)
        self.cumulative_reward = 0
        # While we are not in-game
        while True:
            # Read in current state of switch screen
            ret_val, self.current_frame = self.switch_screen.read()
            if ret_val == False:
                print('Error reading from switch screen. Switch disconnected?')
                print('Exiting')
                exit()
            if RENDER:
                self.render()

            img_resize = cv2.resize(self.current_frame, swap(*INPUT_SHAPE))
            im = Image.fromarray(img_resize)
            im.save('temp.png')
            im_import = cv2.imread('temp.png')

            preprocessed = preprocess(im_import)
            # Evaluate on environment
            env_out = np.argmax(self.env.predict_on_batch(preprocessed))
            if self.DEBUG:
                print('Environment:', CLASS_NAMES_ENV[env_out])
            if env_out == 0:
                # Check if in pregame or GO screen
                pregame_out = np.argmax(self.pregame.predict_on_batch(preprocessed))
                if CLASS_NAMES_PRE[pregame_out] == 'pregame':
                    # Still on pregame, wait
                    if self.DEBUG:
                        print('Waiting on pregame')
                    p_wait(.05)
                    continue
                else:
                    # We have successfully started a new game, return GO screen
                    if self.DEBUG:
                        print('GO')
                    cropped = self.current_frame[:, 480:-480]
                    return cropped
            if env_out == 1 or env_out == 2 or env_out == 3:
                # End game screen, title screen, or error screen
                send_cmd(BTN_A)
            if env_out == 4:
                # Screen for pressing B button
                send_cmd(BTN_B)
            # Send none to clear button press
            p_wait(3.0)
            send_cmd()
            p_wait(3.0)

    def render(self):
        try:
            resized_frame = cv2.resize(self.current_frame, (1280, 720))
        except NameError:
            print('No frames found')
            return
        cv2.imshow('Switch Screen', resized_frame)
        cv2.waitKey(1)
    
    def close(self):
        cv2.destroyAllWindows()
        self.switch_screen.release()
