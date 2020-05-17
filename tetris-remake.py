from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from PIL import Image
import numpy as np
from core import Env
from client-remake import sync, send_cmd, tetris_enum, p_wait, BTN_A, BTN_B, BTN_R, BTN_L, DPAD_D
import cv2
import gym

DEBUG = False

INPUT_SHAPE = (100,100)
OUTPUT_ENV = 5
OUTPUT_DISC = 1
ACTION_SPACE = len(tetris_enum)

CLASS_NAMES = np.array(['neutral', 'end_screen', 'error', 'title_a', 'title_b'])
env_path = 'tetris/environment/env.ckpt'
reward_path = 'tetris/reward/reward.ckpt'

# This file is a wrapper for an OpenAI gym environment
class Tetris(Env):
    action_space = gym.Discrete(ACTION_SPACE)

    def __init__(self, train_disc=False, train_env=False, capture_device=0):
        self.clock = 0
        self.frames = np.zeros(shape=(20, *INPUT_SHAPE))
        self.reward = self.model_reward()
        self.env = self.model_environment()
        # Sync to the switch so that we can send controller commands
        sync()
        if DEBUG:
            print('Successfully synced to MCU')
        # Connect controller
        send_cmd(BTN_L + BTN_R)
        p_wait(0.5)
        send_cmd()
        p_wait(0.1)
        send_cmd(BTN_A)
        p_wait(0.5)
        send_cmd()
        if DEBUG:
            print('Successfully paired controller')
        # Capture switch screen
        try:
            self.switch_screen = cv2.VideoCapture(capture_device)
        except:
            print('Unable to open capture device. Is it plugged in?')
            exit()
        if train_env:
            self.fit_environment()
        self.train_disc = train_disc
        
    # A model that determines what state the game is in
    # (IN_GAME, RESULT_SCREEN, ERROR_SCREEN, TITLE_A, TITLE_B)
    def model_environment(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu')(frame)
        x = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu')(x)
        x = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_ENV, activation='softmax')(x)
        m = Model(inputs = frame, outputs = output)
        try:
            m.load_weights(env_path)
        except:
            print('Unable to load environment network')
            exit()
        return m

    # A model that determines how similar the gameplay is to my gameplay (the good parts)
    # Assigns reward for gameplay
    def model_reward(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu')(frame)
        x = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu')(x)
        x = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_DISC, activation='sigmoid')(x)
        m = Model(inputs = frame, outputs = output)
        try:
            m.load_weights(reward_path)
        except:
            print('Unable to load reward network')
        return m

    # Called every 10 steps. Grabs a random memory and fits against my gameplay
    def fit_reward(self):
        pass

    # Called at the beggining to fit network to training labels (pre-recorded)
    def fit_environment(self, epochs = 20):
        pass

    def step(self, action):
        # Send the chosen button press to switch
        send_cmd(tetris_enum[action])
        p_wait(.05)
        # Read in the resulting frame
        ret_val, self.current_frame = self.switch_screen.read()
        img_resize = K.expand_dims(cv2.resize(img_orig, INPUT_SHAPE), axis=0)
        # Evaluate on environment network
        env_out = np.argmax(self.env(img_resize).numpy()[0])
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
            # in-game
            # Only get reward when placing a block
            if tetris_enum[action] == DPAD_D:
                # Evaluate on reward network
                reward = self.reward(img_resize).numpy()[0][0]
            else:
                reward = 0.0
        self.clock += 1
        return self.current_frame, reward, done, {}

    def reset(self):
        self.clock = 0
        # While we are not in-game
        while True:
            # Read in current state of switch screen
            ret_val, self.current_frame = self.switch_screen.read()
            if ret_val == False:
                print('Error reading from switch screen. Switch disconnected?')
                print('Exiting')
                exit()
            img_resize = K.expand_dims(cv2.resize(img_orig, INPUT_SHAPE), axis=0)
            if DEBUG:
                print('Successfully resized and expanded')
            # Evaluate on environment
            env_out = np.argmax(self.env(img_resize).numpy()[0])
            if DEBUG:
                print('Environment:', env_out)
            if env_out == 0:
                # We have successfully started a new game, return first found frame
                # TO DO optimization: add another classifier to wait until GO screen
                return self.current_frame
            if env_out == 1 or env_out == 2 or env_out == 3:
                # End game screen, title screen, or error screen
                send_cmd(BTN_A)
            if env_out == 4:
                # Screen for pressing B button
                send_cmd(BTN_B)
            # Send none to clear button press
            p_wait(0.1)
            send_cmd()
            p_wait(3.0)

    def render(self):
        try:
            resized_frame = cv2.resize(self.current_frame, (1280, 720))
        except NameError:
            print('No frames found')
            return
        cv2.im_show('Switch Screen', resized_frame)
    
    def close(self):
        cv2.destroyAllWindows()
        self.switch_screen.release()
