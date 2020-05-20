from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from PIL import Image
import numpy as np
from core import Env
from controller import sync, send_cmd, tetris_enum, p_wait, BTN_A, BTN_B, BTN_R, BTN_L, DPAD_U
import cv2
from gym import spaces
import pathlib
from statistics import mean
from random import shuffle, choice

DEBUG = False
RENDER = True

TRAINING_BUFFER = 20
INPUT_SHAPE = (90,160)
OUTPUT_ENV = 5
OUTPUT_DISC = 1
ACTION_SPACE = len(tetris_enum)

CLASS_NAMES = np.array(['neutral', 'end_screen', 'error', 'title_a', 'title_b'])

# Data paths
env_data = pathlib.Path('training/tetris_environment')
reward_data = pathlib.Path('training/tetris_discriminator')

# Model save paths
env_path = 'tetris/environment/env.ckpt'
reward_path = 'tetris/reward/reward.ckpt'

def swap(a,b): return b, a

def one_hot(a, num_classes): return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def preprocess(im): return np.expand_dims(im.astype('float32') / 255.0, axis = 0)

# This file is a wrapper for an OpenAI gym environment
class Tetris(Env):
    action_space = spaces.Discrete(ACTION_SPACE)

    def __init__(self, train_reward=True, train_env=False, capture_device=0):
        self.clock = 0
        self.frames = []
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
            print('Starting environment training')
            self.fit_environment()
            print('Finished environment training - saving')
            print('You can now set train_environment = False')
            self.env.save_weights(env_path)
        self.train_reward = train_reward
        
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
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        try:
            m.load_weights(env_path)
        except:
            print('Unable to load environment network')
        return m

    # A model that determines how similar the gameplay is to my gameplay (the good parts)
    # Assigns reward for gameplay (0 being unsimilar, 1 being similar)
    def model_reward(self):
        frame = Input(shape=(*INPUT_SHAPE, 3))
        x = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu')(frame)
        x = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu')(x)
        x = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(OUTPUT_DISC, activation='sigmoid')(x)
        m = Model(inputs = frame, outputs = output)
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        try:
            m.load_weights(reward_path)
        except:
            print('Unable to load reward network')
        return m

    # Called every 5 steps. Grabs a random memory and fits against my gameplay
    def fit_reward(self):
        batch = np.zeros((2, *INPUT_SHAPE, 3))
        batch[0] = choice(self.frames)
        good_path = str(choice(list(reward_data.glob('clean_data_playing/*/*.png'))))
        batch[1] = preprocess(cv2.imread(good_path))
        labels = np.array([0, 1])
        self.reward.train_on_batch(x = batch, y = labels)
        accuracy = np.mean(self.reward.predict_on_batch(batch) - labels)
        # This value should hopefully approach .5 as the bot learns to play like me
        if DEBUG:
            print('Current reward function accuracy:', accuracy)
    
    # Called at the beggining to fit network to training labels (pre-recorded)
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
                found = np.where(CLASS_NAMES == label)
                labels[x] = one_hot(found[0][0], CLASS_NAMES.size)
                x += 1
                if x == 10:
                    self.env.train_on_batch(x = batch, y = labels)
                    x = 0
        


    def step(self, action):
        # Send the chosen button press to switch
        send_cmd(tetris_enum[action])
        p_wait(.05)
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
        # Unsure if these 2 lines are necessary
        im.save('temp.png')
        im_import = cv2.imread('temp.png')

        preprocessed = preprocess(im_import)

        # Evaluate on environment network
        env_out = np.argmax(self.env.predict_on_batch(preprocessed))
        if DEBUG and self.clock % 10 == 0:
            print('Current environment evaluation:', CLASS_NAMES[env_out])
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
            if tetris_enum[action] == DPAD_U:
                # Evaluate on reward network
                # POSSIBLY MESSED UP HERE
                reward = self.reward.predict_on_batch(preprocessed)[0][0]
                #print(reward)
            else:
                reward = 0.0
        # Add frame to training buffer for reward training
        self.frames.append(preprocessed)
        if len(self.frames) > 20:
            self.frames.pop(0)
        if self.clock % 5 == 0 and self.train_reward:
            self.fit_reward()
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
            if RENDER:
                self.render()

            img_resize = cv2.resize(self.current_frame, swap(*INPUT_SHAPE))
            im = Image.fromarray(img_resize)
            im.save('temp.png')
            im_import = cv2.imread('temp.png')

            preprocessed = preprocess(im_import)
            # Evaluate on environment
            env_out = np.argmax(self.env.predict_on_batch(preprocessed))
            if DEBUG:
                print('Environment:', CLASS_NAMES[env_out])
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
