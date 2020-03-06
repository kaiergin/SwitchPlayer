import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model
from PIL import Image
import pathlib
import numpy as np
import os

#tf.keras.backend.set_floatx('float64')

AUTOTUNE = tf.data.experimental.AUTOTUNE

IM_WIDTH = 160
IM_HEIGHT = 90

SHORT_TERM_BUF = 10
EVAL_FRAMES = 3

OUTPUT_ACTOR = 9
OUTPUT_CRITIC = 1
OUTPUT_ENVIRONMENT = 2

BATCH_SIZE = 1

GAMMA = .99999

checkpoint_path_actor = "tetris/actor/actor.ckpt"
checkpoint_path_critic = "tetris/critic/critic.ckpt"
checkpoint_path_enviornment = "tetris/environment/environment.ckpt"


# Callback for setting checkpoints
# Will resave weights every 5 epochs
cp_callback_actor = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_actor,
        verbose=1,
        save_weights_only=True,
        period=1)

cp_callback_critic = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_critic,
        verbose=1,
        save_weights_only=True,
        period=1)

cp_callback_environment = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_enviornment,
        verbose=1,
        save_weights_only=True,
        period=1)

class Tetris:
    def __init__(self, actor, environment):
        if actor:
            self.previous_moves = tf.zeros((1,SHORT_TERM_BUF,OUTPUT_ACTOR))
            self.previous_inputs = tf.zeros((1,SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
            self.random = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
            self.internal_clock = 0
            self.dataset_built = False
            self.current_save_path = 'training/tetris_actor/game'
            self.actor_opt = tf.keras.optimizers.Adam(1e-4)
            self.critic_opt = tf.keras.optimizers.Adam(1e-4)
            self.actor = self.__create_model_actor()
            self.critic = self.__create_model_critic()
            print("Successfully created actor and critic")
        if environment:
            self.environment = self.__create_model_environment()
            print("Successfully created environment")

    def __create_model_actor(self):
        image_input = Input(shape=(SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        x = layers.Conv3D(120, (3,5,5), activation='relu')(image_input)
        x = layers.MaxPooling3D((2,3,3))(x)
        x = layers.Conv3D(60, (3,5,5), activation='relu')(x)
        x = layers.Flatten()(x)
        move_buffer_input = Input(shape=(SHORT_TERM_BUF,OUTPUT_ACTOR))
        y = layers.Flatten()(move_buffer_input)
        x = layers.concatenate([x,y], axis=1)
        x = layers.Dense(480, activation='relu')(x)
        x = layers.Dense(OUTPUT_ACTOR, activation='sigmoid')(x)

        model = Model(inputs=(image_input, move_buffer_input), outputs=x)

        model.summary()

        '''
        No longer compiling in order to create my own loss function
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        '''
        try:
            model.load_weights(checkpoint_path_actor)
        except:
            print('Error loading weights from file')
        return model
    
    def __create_model_critic(self):
        model = models.Sequential()
        model.add(layers.Conv3D(60, (2, 5, 5), activation='relu', input_shape=(EVAL_FRAMES,IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.AveragePooling3D((1, 3, 3)))
        model.add(layers.Conv3D(30, (2, 5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(OUTPUT_CRITIC, activation='sigmoid'))

        model.summary()

        try:
            model.load_weights(checkpoint_path_critic)
        except:
            print('Error loading weights from file')

        return model

    def __create_model_environment(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.AveragePooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(OUTPUT_ENVIRONMENT, activation='softmax'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        try:
            model.load_weights(checkpoint_path_enviornment)
        except:
            print('Error loading weights from file')
        return model

    # This function will be called every frame to make moves and save results
    def eval_actor(self, image):
        im = Image.fromarray(image)
        im.save(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        img_data = tf.io.read_file(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        im = tf.image.decode_png(img_data, channels=3)
        im = tf.reshape(tf.image.convert_image_dtype(im, tf.float32), (1,1,IM_HEIGHT,IM_WIDTH,3))
        # Roll frame buffer
        self.previous_inputs = tf.slice(self.previous_inputs, (0,0,0,0,0), (1,SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_inputs = tf.concat([im, self.previous_inputs], axis=1)
        # Pass im and buffer data to get next move probabilities
        output = self.actor((self.previous_inputs, self.previous_moves))
        # Create random (0,1) tensor the same size as output
        random_tensor = self.random(shape=(1,OUTPUT_ACTOR))
        # Compare > to get which moves fired
        result = tf.math.greater(output, random_tensor)
        # Convert from boolean tensor to float tensor
        result_input_tensor = tf.cast(tf.equal(tf.reshape(result, (1,1,OUTPUT_ACTOR)), True), tf.float32)
        result_input_file = tf.cast(tf.equal(tf.reshape(result, (1,OUTPUT_ACTOR)), True), tf.float32).numpy()
        # Load move file
        try:
            move_tensor = np.load(self.current_save_path + '/0.npy')
            move_tensor = np.concatenate((move_tensor, result_input_file), axis=0)
        except FileNotFoundError:
            move_tensor = result_input_file
        # Roll move buffer
        self.previous_moves = tf.slice(self.previous_moves, (0,0,0), (1,SHORT_TERM_BUF-1,OUTPUT_ACTOR))
        self.previous_moves = tf.concat([result_input_tensor, self.previous_moves], axis=1)
        # Save move for current frame
        np.save(self.current_save_path + '/0.npy', move_tensor)
        # Update clock
        self.internal_clock += 1
        # Return chosen moves
        return result.numpy()

    # This function is used to determine if a game of tetris was lost
    def eval_environment(self, image):
        return self.environment(image).numpy()

    def fit_enviornment(self, input_generator, spe, epochs=100):
        self.environment.fit(input_generator, 
                    epochs = epochs,
                    verbose = 1,
                    steps_per_epoch = spe,
                    callbacks = [cp_callback_environment])
    
    def __build_dataset(self):
        if self.dataset_built:
            return
        self.dataset_built = True
        data_path = pathlib.Path(self.current_save_path)
        self.image_count = len(list(data_path.glob("*.png")))
        if self.image_count < 100:
            print("Game too short, cannot build dataset")
            self.dataset_built = False
            return
        self.cur_index = self.image_count - (EVAL_FRAMES + 1) # start from the back of the dataset
        self.cur_discount = -1 * GAMMA
        self.move_tensor = np.load(self.current_save_path + '/0.npy').astype('float32')
    
    def __get_next_batch(self):
        def build_path(name):
            return self.current_save_path + "/" + name + ".png"
        def process_path(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_png(img)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.expand_dims(img, axis=0)
            return img
        images = []
        for x in range(SHORT_TERM_BUF):
            path = build_path(str(self.cur_index - x))
            images.append(process_path(path))
        image_buf = tf.concat(images, axis=0)
        image_buf = tf.expand_dims(image_buf, axis=0)
        # Get the resulting frames from doing move
        images = []
        for x in range(EVAL_FRAMES):
            path = build_path(str(self.cur_index + x))
            images.append(process_path(path))
        result_buf = tf.concat(images, axis=0)
        result_buf = tf.expand_dims(result_buf, axis=0)
        move_chosen = self.move_tensor[self.cur_index]
        # The move buffer does not have the move for the
        # current frame (cur_index) so we go from from cur_index - (SHORT_TERM_BUF)
        move_buf = self.move_tensor[self.cur_index-(SHORT_TERM_BUF):self.cur_index]
        move_buf = tf.convert_to_tensor(move_buf)
        move_buf = tf.expand_dims(move_buf, axis=0)
        self.cur_discount *= GAMMA
        return image_buf, move_buf, move_chosen, result_buf

    # This function will be called after a game of tetris is lost
    def fit_actor(self):
        self.__build_dataset()
        it = 0
        while (self.cur_index-(SHORT_TERM_BUF+1) > 0):
            batch = self.__get_next_batch()
            self.__train_step(batch)
            self.cur_index -= EVAL_FRAMES
        self.__reset_actor()

    def __reset_actor(self):
        self.previous_moves = tf.zeros((1,SHORT_TERM_BUF,OUTPUT_ACTOR))
        self.previous_inputs = tf.zeros((1,SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        self.internal_clock = 0
        self.dataset_built = False
        # Clear folder
        for filename in os.listdir(self.current_save_path):
            file_path = os.path.join(self.current_save_path, filename)
            os.unlink(file_path)
    
    @tf.function
    def __train_step(self, batch):
        image_buf, move_buf, move_chosen, result_buf = batch
        with tf.GradientTape() as act_tape, tf.GradientTape() as crit_tape:
            probs = self.actor((image_buf, move_buf), training=True)
            reward = self.critic(result_buf, training=True)
            advantage = (self.cur_discount + 1) - reward
            log_probs = tf.math.log(probs)
            actor_loss = -1 * log_probs * advantage
            critic_loss = 0.5 * tf.pow(advantage,2)
        gradients_of_actor = act_tape.gradient(actor_loss, self.actor.trainable_variables)
        gradients_of_critic = crit_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))

