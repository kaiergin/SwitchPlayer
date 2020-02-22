import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanAbsoluteError
from PIL import Image
import os

tf.keras.backend.set_floatx('float64')

IM_WIDTH = 160
IM_HEIGHT = 90

SHORT_TERM_BUF = 10
MAX_GAMES_SAVED = 5

OUTPUT_CRITIC = 2
OUTPUT_ACTOR = 10

checkpoint_path = "tetris/actor.ckpt"
checkpoint_path_critic = "tetris/critic.ckpt"

# Callback for setting checkpoints
# Will resave weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=1)

cp_callback_critic = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_critic,
        verbose=1,
        save_weights_only=True,
        period=1)

class Tetris:
    def __init__(self, actor, critic):
        self.previous_moves = tf.zeros((SHORT_TERM_BUF,OUTPUT_ACTOR))
        self.previous_inputs = tf.zeros((SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        #self.internal_state = tf.zeros(20)
        self.random = tf.random_uniform_initializer(minval=0.0, maxval=1.0, shape=OUTPUT_ACTOR)
        self.internal_clock = 0
        self.current_save_path = 'training/tetris_actor/game'
        if actor:
            self.actor = self.__create_model_input()
            print("Successfully created actor")
        if critic:
            self.critic = self.__create_model_critic()
            print("Successfully created critic")

    def gen_actor(self):
        self.actor = self.__create_model_input()
        print("Successfully created actor")

    def gen_critic(self):
        self.critic = self.__create_model_critic()
        print("Successfully created critic")

    def __create_model_input(self):
        image_input = Input(shape=(SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        x = layers.Conv2D(image_input, 60, (5,5), activation='relu')
        x = layers.MaxPooling2D(x, (3,3))
        x = layers.Conv2D(image_input, 60, (5,5), activation='relu')
        x = layers.Flatten(x)
        move_buffer_input = Input(shape=(SHORT_TERM_BUF,OUTPUT_ACTOR))
        y = layers.Flatten(move_buffer_input)
        x = layers.concatenate((x,y))
        x = layers.Dense(x, 240, activation='relu')
        x = layers.Dense(x, OUTPUT_ACTOR, activation='sigmoid')

        model = Model(inputs=(image_input, move_buffer_input), outputs=x)
        '''
        model = models.Sequential()
        model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(60, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))


        model.add(layers.Flatten())
        model.add(layers.Dense(240, activation='relu'))
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(OUTPUT_ACTOR, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        '''
        try:
            model.load_weights(checkpoint_path)
        except:
            print('Error loading weights from file')
        return model

    def __create_model_critic(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.AveragePooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(OUTPUT_CRITIC, activation='softmax'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        try:
            model.load_weights(checkpoint_path_critic)
        except:
            print('Error loading weights from file')
        return model

    def eval_actor(self, image):
        im = Image.fromarray(image)
        im.save(self.current_save_path + os.path.sep + str(self.internal_clock))
        img_data = tf.io.read_file(self.current_save_path + os.path.sep + str(self.internal_clock))
        im = tf.image.decode_png(img_data, channels=3)
        im = tf.reshape(tf.image.convert_image_dtype(im, tf.float64), (1,IM_HEIGHT,IM_WIDTH,3))
        # Roll frame buffer
        self.previous_inputs = tf.slice(self.previous_inputs, (0,0,0,0), (SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_inputs = tf.concat(im, self.previous_inputs, axis=0)
        # Pass im and buffer data to get next move probabilities
        output = self.critic((self.previous_inputs, self.previous_moves)).numpy()
        # Create random tensor size of next move probabilities
        random_tensor = self.random()
        # Compare > to get which moves fired
        result = tf.math.greater(output, random_tensor)
        # Roll move buffer
        self.previous_moves = tf.slice(self.previous_moves, (0,0,0), (SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_inputs = tf.concat(result, self.previous_inputs, axis=0)
        # Update clock
        self.internal_clock += 1
        # Return chosen moves
        return result.numpy()

    def eval_critic(self, image):
        return self.critic(image).numpy()

    def fit_critic(self, input_generator, spe, epochs=100):
        self.critic.fit(input_generator, epochs = epochs, verbose=1, steps_per_epoch = spe, callbacks = [cp_callback_critic])
