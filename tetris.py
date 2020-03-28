import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input, Model
from PIL import Image
import pathlib
import numpy as np
import os
import random

#tf.keras.backend.set_floatx('float64')

AUTOTUNE = tf.data.experimental.AUTOTUNE

IM_WIDTH = 160
IM_HEIGHT = 90

OUTPUT_ACTOR_DPAD = 4
OUTPUT_ACTOR_OTHR = 4
OUTPUT_ACTOR = OUTPUT_ACTOR_DPAD + OUTPUT_ACTOR_OTHR
OUTPUT_CRITIC = 1
OUTPUT_ENVIRONMENT = 5

BATCH_SIZE = 1

GAMMA = .95

TRAIN_DISC = False
TD_LEARNING = True
RANDOM_SELECTION_LEARNING = True
EPSILON_LEARNING = False
RANDOM_NOISE_LEARNING = False

EPSILON_GROWTH = .003
EPSILON_MAX = .95
EPSILON_START = .1

data_track = "actor_critic_loss.csv"
checkpoint_path_actor = "tetris/actor/actor.ckpt"
checkpoint_path_critic = "tetris/critic/critic.ckpt"
checkpoint_path_enviornment = "tetris/environment/environment.ckpt"
checkpoint_path_discriminator = "tetris/discriminator/discriminator.ckpt"
checkpoint_path_tetris_finder = "tetris/finder/finder.ckpt"

cp_callback_environment = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_enviornment,
        verbose=1,
        save_weights_only=True,
        period=1)

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def get_discriminator_image():
    working_dir = 'training/tetris_discriminator/'
    images = list(pathlib.Path(working_dir).glob('bad_game_*/*.png'))
    image_chosen = random.choice(images)
    good_image = process_path(str(image_chosen))
    return good_image

class Tetris:
    def __init__(self, actor, environment):
        if actor:
            self.random = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
            self.epsilon = EPSILON_START
            self.current_save_path = 'training/tetris_actor/game'
            self.actual_future_reward = GAMMA
            # Create optimizers
            self.actor_opt = tf.keras.optimizers.Adam(1e-7)
            self.critic_opt = tf.keras.optimizers.Adam(1e-5)
            self.disc_opt = tf.keras.optimizers.Adam(1e-5)
            # Create models
            self.actor = self.__create_model_actor()
            self.critic = self.__create_model_critic()
            self.discriminator = self.__create_model_discriminator()
             # Create networks to edit and save
            if not TD_LEARNING:
                self.edit_actor = self.__create_model_actor()
            # Sets states of actor and clears save folder
            self.reset_actor()
            print("Successfully created actor and critic")
        if environment:
            self.environment = self.__create_model_environment()
            # self.tetris_finder = self.__create_tetris_finder()
            print("Successfully created environment")

    # Evaluates a frame of tetris to get move probabilities
    def __create_model_actor(self):
        image_input = Input(shape=(IM_HEIGHT,IM_WIDTH,3))
        x = layers.Conv2D(60, (10, 10), activation='relu')(image_input)
        x = layers.MaxPooling2D((3, 3))(x)
        x = layers.Conv2D(30, (5, 5), activation='relu')(x)
        x = layers.Flatten()(x)
        #move_buffer_input = Input(shape=(SHORT_TERM_BUF,OUTPUT_ACTOR))
        #y = layers.Flatten()(move_buffer_input)
        #x = layers.concatenate([x,y], axis=1)
        x = layers.Dense(120, activation='relu')(x)
        x = layers.Dense(OUTPUT_ACTOR, activation='softmax')(x)
        #x = layers.Dense(OUTPUT_ACTOR_DPAD, activation='softmax')(x)
        #y = layers.Dense(OUTPUT_ACTOR_OTHR, activation='softmax')(x)

        model = Model(inputs=image_input, outputs=x)

        model.summary()

        try:
            model.load_weights(checkpoint_path_actor)
        except:
            print('Error loading weights from file')
        return model
    
    def __create_model_critic(self):
        image_input = Input(shape=(IM_HEIGHT,IM_WIDTH,3))
        x = layers.Conv2D(60, (10, 10), activation='relu')(image_input)
        x = layers.MaxPooling2D((3, 3))(x)
        x = layers.Conv2D(30, (5, 5), activation='relu')(x)
        x = layers.Flatten()(x)
        move_input = Input(shape=(OUTPUT_ACTOR))
        x = layers.concatenate([x,move_input], axis=1)
        x = layers.Dense(120, activation='relu')(x)
        x = layers.Dense(OUTPUT_CRITIC)(x)

        model = Model(inputs=(image_input,move_input), outputs=x)

        model.summary()

        try:
            model.load_weights(checkpoint_path_critic)
        except:
            print('Error loading weights from file')

        return model

    def __create_model_environment(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (10, 10), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
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
    
    def __create_model_discriminator(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (10, 10), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(OUTPUT_CRITIC, activation='sigmoid'))
        model.summary()

        try:
            model.load_weights(checkpoint_path_discriminator)
        except:
            print('Error loading weights from file')

        return model
    
    def __create_tetris_finder(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (10, 10), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))
        model.summary()

        try:
            model.load_weights(checkpoint_path_tetris_finder)
        except:
            print('Error loading weights from file')

        return model

    # This function will be called every frame to make moves and save results
    def eval_actor(self, image):
        im = Image.fromarray(image)
        im.save(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        img_data = tf.io.read_file(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        im = tf.image.decode_png(img_data, channels=3)
        im = tf.reshape(tf.image.convert_image_dtype(im, tf.float32), (1,IM_HEIGHT,IM_WIDTH,3))
        # Get button probabilities
        output_btn = tf.squeeze(self.actor(im))
        # Reward for current frame
        disc_cur = tf.squeeze(self.discriminator(im))
        # Estimated future reward for previous move 
        crit_cur = tf.squeeze(self.critic((self.previous_frame, self.previous_move)))
        if EPSILON_LEARNING:
            random_num = self.random([1]).numpy()[0]
            # If random number is greater than epsilon, take random path, else choose max
            if random_num > self.epsilon:
                chosen_btn = tf.one_hot(tf.math.argmax(tf.random.uniform([OUTPUT_ACTOR])), OUTPUT_ACTOR)
            else:
                chosen_btn = tf.one_hot(tf.math.argmax(output_btn), OUTPUT_ACTOR)
        elif RANDOM_SELECTION_LEARNING:
            # Selects based on output probability
            chosen_btn = tf.one_hot(tf.random.categorical(tf.math.log([output_btn]), 1), OUTPUT_ACTOR)
        chosen_btn = tf.reshape(chosen_btn, (1,OUTPUT_ACTOR))
        result_input_file = tf.cast(chosen_btn, tf.float32).numpy()
        
        # Save state, action, next state, next move for realtime training
        self.batch = (self.previous_frame, self.previous_move, im, chosen_btn)
        
        if self.internal_clock % 50 == 0:
            print('Button distribution', output_btn.numpy())
            print('Chosen button', chosen_btn.numpy())
            print('Discriminator output', disc_cur.numpy())
            print('Critic output', crit_cur.numpy())

        # Load move file
        try:
            move_tensor = np.load(self.current_save_path + '/moves.npy')
            move_tensor = np.concatenate((move_tensor, result_input_file), axis=0)
        except FileNotFoundError:
            move_tensor = result_input_file
        # Update previous frame
        self.previous_frame = im
        # Update previous move
        self.previous_move = chosen_btn
        # Save move for current frame
        np.save(self.current_save_path + '/moves.npy', move_tensor)
        # Update clock
        self.internal_clock += 1
        # Return chosen moves
        return tf.squeeze(chosen_btn).numpy()

    # This function is used to determine if a game of tetris was lost
    def eval_environment(self, image):
        return self.environment(image).numpy()

    def fit_enviornment(self, input_generator, spe, epochs=100):
        self.environment.fit(input_generator, 
                    epochs = epochs,
                    verbose = 1,
                    steps_per_epoch = spe,
                    callbacks = [cp_callback_environment])
    
    '''
    def eval_tetris_finder(self, image):
        return self.tetris_finder(image).numpy()
    
    def fit_tetris_finder(self, input_generator, spe, epochs=100):
        self.tetris_finder.fit(input_generator, 
                    epochs = epochs,
                    verbose = 1,
                    steps_per_epoch = spe,
                    callbacks = [cp_callback_environment])
    '''

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
        self.cur_index = self.image_count - 2 # start from the back of the dataset
        self.move_tensor = np.load(self.current_save_path + '/moves.npy').astype('float32')
    
    def __get_next_batch(self):
        def build_path(name):
            return self.current_save_path + "/" + name + ".png"
        state = process_path(build_path(str(self.cur_index)))
        # Get the resulting frames from doing move
        next_state = state = process_path(build_path(str(self.cur_index + 1)))
        # Get move chosen
        move_chosen = np.reshape(self.move_tensor[self.cur_index], (1, OUTPUT_ACTOR))
        # Gets a random image from game of a tetris player knows what they're doing
        good_image = get_discriminator_image()
        return state, move_chosen, next_state, good_image
    
    def fit_actor_realtime(self):
        if not TD_LEARNING:
            return
        a = self.__train_step_realtime(self.batch)
        if self.internal_clock % 50 == 0:
            print('Actor loss:', a)

    # This function will be called after a game of tetris is lost
    def fit_actor(self):
        actor_loss, critic_loss, disc_loss = 0, 0, 0
        it = 0
        self.__build_dataset()
        if not self.dataset_built:
            self.reset_actor()
            return
        # Q value for losing
        qval = -20
        while (self.cur_index > 1):
            batch = self.__get_next_batch()
            qval, a, c, d = self.__train_step(qval, batch)
            actor_loss += a
            critic_loss += c
            disc_loss += d
            self.cur_index -= 1
            it += 1
        # Record losses
        actor_loss = actor_loss.numpy()
        critic_loss = critic_loss.numpy()
        disc_loss = disc_loss.numpy()
        with open(data_track, 'a+') as f:
            f.write(str(actor_loss) + ',' + str(critic_loss) + ',' + str(disc_loss) + '\n')
            print('Actor loss:', actor_loss, 'Critic loss:', critic_loss, 'Discriminator loss:', disc_loss, 'Num moves:', it)
        # Increase epsilon after each game
        self.epsilon += (1 - self.epsilon) * EPSILON_GROWTH
        self.epsilon = min(self.epsilon, EPSILON_MAX)
        if EPSILON_LEARNING:
            print('Current epsilon:', str(self.epsilon))
        # Save current state of each neural networks
        self.critic.save_weights(checkpoint_path_critic)
        if TRAIN_DISC:
            self.discriminator.save_weights(checkpoint_path_discriminator)
        if TD_LEARNING:
            self.actor.save_weights(checkpoint_path_actor)
        else:
            self.edit_actor.save_weights(checkpoint_path_actor)
            self.actor.load_weights(checkpoint_path_actor)
        self.reset_actor()

    def reset_actor(self):
        self.previous_frame = tf.zeros((1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_move = self.actor(self.previous_frame)
        self.internal_clock = 0
        self.dataset_built = False
        # Clear folder
        for filename in os.listdir(self.current_save_path):
            file_path = os.path.join(self.current_save_path, filename)
            os.unlink(file_path)

    # Only for editing actor
    @tf.function
    def __train_step_realtime(self, batch):
        state, move_chosen, next_state, next_move = batch
        with tf.GradientTape() as act_tape:
            act_tape.watch(self.actor.trainable_variables)
            # Evaluate on models
            probs = tf.squeeze(self.actor(state))
            V_cur = tf.squeeze(self.critic((state, move_chosen)))
            V_future = tf.squeeze(self.critic((next_state, next_move)))
            # Currently only calculated by discriminator value (will add ingame rewards such as lineclears)
            reward = tf.squeeze(self.discriminator(next_state)) # + 5 * tf.squeeze(self.find_tetris(result_buf[1]))[1]
            # Calculate TD advantage
            advantage = reward + GAMMA*V_future - V_cur
            log_probs = tf.math.log(probs)
            actor_loss = -1 * log_probs * advantage
        gradients_of_actor = act_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        return actor_loss

    # Updates all networks (except for actor if REALTIME_TRAINING is True)
    @tf.function
    def __train_step(self, qval, batch):
        state, move_chosen, next_state, good_image = batch
        with tf.GradientTape() as act_tape, tf.GradientTape() as crit_tape, tf.GradientTape() as disc_tape:
            # Assign tape to variables
            act_tape.watch(self.actor.trainable_variables)
            crit_tape.watch(self.critic.trainable_variables)
            disc_tape.watch(self.discriminator.trainable_variables)
            # Evaluate on models
            probs = tf.squeeze(self.actor(state))
            V_cur = tf.squeeze(self.critic((state,move_chosen)))
            disc_future = tf.squeeze(self.discriminator(next_state))
            disc_good = tf.squeeze(self.discriminator(good_image))
            reward = disc_future # Reward currently only calculated by discriminator
            advantage = reward + GAMMA*qval - V_cur
            newq = disc_future + GAMMA*qval
            log_probs = tf.math.log(probs)
            actor_loss = -1 * log_probs * advantage
            critic_loss = tf.math.square(advantage)
            disc_loss = disc_future + (1 - disc_good)
        if not TD_LEARNING:
            gradients_of_actor = act_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(gradients_of_actor, self.edit_actor.trainable_variables))
        gradients_of_critic = crit_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        if TRAIN_DISC:
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return newq, actor_loss, critic_loss, disc_loss
