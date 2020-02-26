import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanAbsoluteError
from PIL import Image
import pathlib
import numpy as np
import os

tf.keras.backend.set_floatx('float64')

AUTOTUNE = tf.data.experimental.AUTOTUNE

IM_WIDTH = 160
IM_HEIGHT = 90

SHORT_TERM_BUF = 10
MAX_GAMES_SAVED = 5

OUTPUT_CRITIC = 2
OUTPUT_ACTOR = 9

BATCH_SIZE = 1
EPOCHS = SHORT_TERM_BUF - 1

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
        self.previous_moves = tf.zeros((SHORT_TERM_BUF - 1,OUTPUT_ACTOR))
        self.previous_inputs = tf.zeros((SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        #self.internal_state = tf.zeros(20)
        self.random = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
        self.internal_clock = 0
        self.current_save_path = 'training/tetris_actor/game'
        if actor:
            self.actor = self.__create_model_actor()
            print("Successfully created actor")
        if critic:
            self.critic = self.__create_model_critic()
            print("Successfully created critic")

    def __create_model_actor(self):
        image_input = Input(shape=(SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        x = layers.ConvLSTM2D(image_input, 60, (5,5), activation='relu')
        x = layers.MaxPooling2D(x, (3,3))
        x = layers.ConvLSTM2D(x, 60, (5,5), activation='relu')
        x = layers.Flatten(x)
        move_buffer_input = Input(shape=(SHORT_TERM_BUF,OUTPUT_ACTOR))
        y = layers.Flatten(move_buffer_input)
        x = layers.concatenate((x,y), axis=0)
        x = layers.Dense(x, 480, activation='relu')
        x = layers.Dense(x, OUTPUT_ACTOR, activation='sigmoid')

        model = Model(inputs=(image_input, move_buffer_input), outputs=x)

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
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

    # This function will be called every frame to make moves and save results
    def eval_actor(self, image):
        im = Image.fromarray(image)
        im.save(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        img_data = tf.io.read_file(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        im = tf.image.decode_png(img_data, channels=3)
        im = tf.reshape(tf.image.convert_image_dtype(im, tf.float64), (1,IM_HEIGHT,IM_WIDTH,3))
        # Roll frame buffer
        self.previous_inputs = tf.slice(self.previous_inputs, (0,0,0,0), (SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_inputs = tf.concat(im, self.previous_inputs, axis=0)
        # Pass im and buffer data to get next move probabilities
        output = self.critic((self.previous_inputs, self.previous_moves)).numpy()
        # Create random (0,1) tensor the same size as output
        random_tensor = self.random(shape=(1,OUTPUT_ACTOR))
        # Compare > to get which moves fired
        result = tf.math.greater(output, random_tensor)
        # Roll move buffer
        self.previous_moves = tf.slice(self.previous_moves, (0,0,0), (SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_inputs = tf.concat(result, self.previous_inputs, axis=0)
        # Save move for current frame
        tf.io.write_file(self.current_save_path + '/' + str(self.internal_clock) + '.tns', result)
        # Update clock
        self.internal_clock += 1
        # Return chosen moves
        return result.numpy()

    # This function is used to determine if a game of tetris was lost
    def eval_critic(self, image):
        return self.critic(image).numpy()

    def fit_critic(self, input_generator, spe, epochs=100):
        self.critic.fit(input_generator, 
                    epochs = epochs,
                    verbose = 1,
                    steps_per_epoch = spe,
                    callbacks = [cp_callback_critic])
    
    # This function will be called after a game of tetris is lost
    def fit_actor(self):
        data_path = pathlib.Path(self.current_save_path)
        list_train_ds = tf.data.Dataset.list_files(str(data_path/'*.png'), shuffle=False)
        # Each timestamp has a corresponding move file, made at the same timestamp
        list_train_moves_ds = tf.data.Dataset.list_files(str(data_path/'*.tns'))
        image_count = len(list(data_path.glob("*.png")))
        def process_path(file_path):
            img = tf.io.read_file(file_path)
            img = tf.decode_png(img)
            return img
        train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        train_move_ds = list_train_moves_ds.map(tf.io.read_file, num_parallel_calls=AUTOTUNE)
        def prepare_for_training(skip, ds):
            # Uses skip to offset data for each epoch so that model is training on different predictions
            ds.skip(skip)
            # Batches into frames of 10
            ds.batch(SHORT_TERM_BUF, drop_remainder=True)
            # Reverse because neural network takes in from most recent first
            a = lambda x: tf.reverse(x, 0)
            ds.map(a, num_parallel_calls=AUTOTUNE)
            return ds
        for x in range(EPOCHS):
            # Images must be offset by +1 due to moves being saved after image on same timestep
            ds_images = prepare_for_training(x+1, train_ds)
            ds_moves = prepare_for_training(x, train_move_ds)
            ds = tf.data.Dataset.zip(ds_moves, ds_images)
            # ASSIGN LABELS HERE
            # TO DO
            ds.batch(BATCH_SIZE)
            self.actor.fit(ds, epochs=1)

    def reset_actor(self):
        self.previous_moves = tf.zeros((SHORT_TERM_BUF,OUTPUT_ACTOR))
        self.previous_inputs = tf.zeros((SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        self.internal_clock = 0
        self.actor.reset_states()
        # Clear folder
        for filename in os.listdir(self.current_save_path):
            file_path = os.path.join(self.current_save_path, filename)
            os.unlink(file_path)

