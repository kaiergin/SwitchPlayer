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

# Replay buffer size
SHORT_TERM_BUF = 3
# Number of frames to evaluate on (including current frame)
EVAL_FRAMES = 2

OUTPUT_ACTOR_DPAD = 4
OUTPUT_ACTOR_OTHR = 4
OUTPUT_ACTOR = OUTPUT_ACTOR_DPAD + OUTPUT_ACTOR_OTHR
OUTPUT_CRITIC = 1
OUTPUT_ENVIRONMENT = 5

BATCH_SIZE = 1

GAMMA = .95

REALTIME_LEARNING = True
EPSILON_LEARNING = True

EPSILON_GROWTH = .001
EPSILON_MAX = .95
EPSILON_START = .1

data_track = "actor_critic_loss.csv"
checkpoint_path_actor = "tetris/actor/actor.ckpt"
checkpoint_path_critic = "tetris/critic/critic.ckpt"
checkpoint_path_enviornment = "tetris/environment/environment.ckpt"
checkpoint_path_discriminator = "tetris/discriminator/discriminator.ckpt"

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
            self.actor_opt = tf.keras.optimizers.Adam(1e-8)
            self.critic_opt = tf.keras.optimizers.Adam(1e-5)
            self.disc_opt = tf.keras.optimizers.Adam(1e-5)
            # Create models
            self.actor = self.__create_model_actor()
            self.critic = self.__create_model_critic()
            self.discriminator = self.__create_model_discriminator()
             # Create networks to edit and save
            if not REALTIME_LEARNING:
                self.edit_actor = self.__create_model_actor()
            # Sets states of actor and clears save folder
            self.reset_actor()
            print("Successfully created actor and critic")
        if environment:
            self.environment = self.__create_model_environment()
            print("Successfully created environment")

    # Change to CRNN
    def __create_model_actor(self):
        image_input = Input(shape=(SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        x = layers.Conv3D(60, (1, 10, 10), activation='relu')(image_input)
        x = layers.MaxPooling3D((1, 3, 3))(x)
        x = layers.Conv3D(30, (1, 5, 5), activation='relu')(x)
        x = layers.Flatten()(x)
        move_buffer_input = Input(shape=(SHORT_TERM_BUF,OUTPUT_ACTOR))
        y = layers.Flatten()(move_buffer_input)
        x = layers.concatenate([x,y], axis=1)
        x = layers.Dense(480, activation='relu')(x)
        x = layers.Dense(OUTPUT_ACTOR, activation='softmax')(x)
        #x = layers.Dense(OUTPUT_ACTOR_DPAD, activation='softmax')(x)
        #y = layers.Dense(OUTPUT_ACTOR_OTHR, activation='softmax')(x)

        model = Model(inputs=(image_input, move_buffer_input), outputs=x)

        model.summary()

        try:
            model.load_weights(checkpoint_path_actor)
        except:
            print('Error loading weights from file')
        return model
    
    def __create_model_critic(self):
        model = models.Sequential()
        model.add(layers.Conv2D(60, (10, 10), activation='relu', input_shape=(IM_HEIGHT,IM_WIDTH,3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

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

    # This function will be called every frame to make moves and save results
    def eval_actor(self, image):
        im = Image.fromarray(image)
        im.save(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        img_data = tf.io.read_file(self.current_save_path + '/' + str(self.internal_clock) + '.png')
        im = tf.image.decode_png(img_data, channels=3)
        im = tf.reshape(tf.image.convert_image_dtype(im, tf.float32), (1,1,IM_HEIGHT,IM_WIDTH,3))
        im_dsc = tf.reshape(tf.image.convert_image_dtype(im, tf.float32), (1,IM_HEIGHT,IM_WIDTH,3))
        good_image = get_discriminator_image()
        result_buf = [tf.slice(tf.squeeze(self.previous_frames), (0,0,0,0), (1,IM_HEIGHT,IM_WIDTH,3)), im_dsc]
        self.batch = (self.prev_state[0], self.prev_state[1], self.previous_move, result_buf, good_image)
        self.prev_state[0] = tf.identity(self.previous_frames)
        self.prev_state[1] = tf.identity(self.previous_moves)
        # Roll frame buffer
        self.previous_frames = tf.slice(self.previous_frames, (0,0,0,0,0), (1,SHORT_TERM_BUF-1,IM_HEIGHT,IM_WIDTH,3))
        self.previous_frames = tf.concat([im, self.previous_frames], axis=1)
        # Pass im and buffer data to get next move probabilities
        # output_dir, output_btn = tf.squeeze(self.actor((self.previous_frames, self.previous_moves)))
        output_btn = tf.squeeze(self.actor((self.previous_frames, self.previous_moves)))
        disc_cur = tf.squeeze(self.discriminator(im_dsc))
        crit_cur = tf.squeeze(self.critic(im_dsc))
        if EPSILON_LEARNING:
            random_num = self.random([1]).numpy()[0]
            # If random number is greater than epsilon, take random path, else choose max
            if random_num > self.epsilon:
                chosen_btn = tf.squeeze(tf.one_hot(tf.math.argmax(tf.random.uniform([OUTPUT_ACTOR])), OUTPUT_ACTOR))
            else:
                chosen_btn = tf.squeeze(tf.one_hot(tf.math.argmax(output_btn), OUTPUT_ACTOR))
        else:
            # chosen_dir = tf.squeeze(tf.one_hot(tf.random.categorical(tf.math.log([output_dir]), 1), OUTPUT_ACTOR_DPAD))
            chosen_btn = tf.squeeze(tf.one_hot(tf.random.categorical(tf.math.log([output_btn]), 1), OUTPUT_ACTOR))
        # result = tf.concat([chosen_dir, chosen_btn], axis=0)
        result = chosen_btn
        result_input_file = tf.reshape(tf.cast(result, tf.float32), (1, OUTPUT_ACTOR)).numpy()
        result_input_tensor = tf.reshape(result, (1,1,OUTPUT_ACTOR))

        if self.internal_clock % 50 == 0:
            #print('Direction distribution', output_dir.numpy())
            print('Button distribution', output_btn.numpy())
            #print('Chosen direction', chosen_dir.numpy())
            print('Chosen button', chosen_btn.numpy())
            print('Discriminator output', disc_cur.numpy())
            print('Critic output', crit_cur.numpy())

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
        self.previous_move = chosen_btn.numpy()
        # Return chosen moves
        return chosen_btn.numpy()

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
        self.actual_future_reward = GAMMA
        self.move_tensor = np.load(self.current_save_path + '/0.npy').astype('float32')
    
    def __get_next_batch(self):
        def build_path(name):
            return self.current_save_path + "/" + name + ".png"
        images = []
        for x in range(SHORT_TERM_BUF):
            path = build_path(str(self.cur_index - x))
            images.append(process_path(path))
        image_buf = tf.concat(images, axis=0)
        image_buf = tf.expand_dims(image_buf, axis=0)
        # Get the resulting frames from doing move
        result_buf = []
        for x in range(EVAL_FRAMES):
            path = build_path(str(self.cur_index + x))
            result_buf.append(process_path(path))
        move_chosen = self.move_tensor[self.cur_index]
        # The move buffer does not have the move for the
        # current frame (cur_index) so we go from from cur_index - SHORT_TERM_BUF
        move_buf = self.move_tensor[self.cur_index - SHORT_TERM_BUF:self.cur_index]
        move_buf = tf.convert_to_tensor(move_buf)
        move_buf = tf.expand_dims(move_buf, axis=0)
        # Gets a random image from game of a tetris player knows what they're doing
        good_image = get_discriminator_image()
        return image_buf, move_buf, move_chosen, result_buf, good_image
    
    def fit_actor_realtime(self):
        if not REALTIME_LEARNING:
            return
        a, c, d = self.__train_step(self.batch)
        if self.internal_clock % 50 == 0:
            print('Actor loss:', a, 'Critic loss:', c, 'Discriminator loss:', d)

    # This function will be called after a game of tetris is lost
    def fit_actor(self):
        actor_loss, critic_loss, disc_loss = 0, 0, 0
        it = 0
        if not REALTIME_LEARNING:
            self.__build_dataset()
            if not self.dataset_built:
                self.reset_actor()
                return
            while (self.cur_index-(SHORT_TERM_BUF+1) > 0):
                batch = self.__get_next_batch()
                a, c, d = self.__train_step(batch)
                actor_loss += a
                critic_loss += c
                disc_loss += d
                self.cur_index -= 1
                it += 1
                self.actual_future_reward *= GAMMA
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
        # Save current state of each neural network
        self.critic.save_weights(checkpoint_path_critic)
        self.discriminator.save_weights(checkpoint_path_discriminator)
        if REALTIME_LEARNING:
            self.actor.save_weights(checkpoint_path_actor)
        else:
            self.edit_actor.save_weights(checkpoint_path_actor)
            self.actor.load_weights(checkpoint_path_actor)
        self.reset_actor()

    def reset_actor(self):
        self.previous_moves = tf.zeros((1,SHORT_TERM_BUF,OUTPUT_ACTOR))
        self.previous_frames = tf.zeros((1,SHORT_TERM_BUF,IM_HEIGHT,IM_WIDTH,3))
        self.prev_state = [tf.identity(self.previous_frames), tf.identity(self.previous_moves)]
        self.previous_move = tf.squeeze(self.actor((self.previous_frames, self.previous_moves))).numpy()
        self.internal_clock = 0
        self.dataset_built = False
        # Clear folder
        for filename in os.listdir(self.current_save_path):
            file_path = os.path.join(self.current_save_path, filename)
            os.unlink(file_path)
    
    @tf.function
    def __train_step(self, batch):
        image_buf, move_buf, move_chosen, result_buf, good_image = batch
        with tf.GradientTape() as act_tape, tf.GradientTape() as crit_tape, tf.GradientTape() as disc_tape:
            # Assign tape to variables
            act_tape.watch(self.actor.trainable_variables)
            crit_tape.watch(self.critic.trainable_variables)
            disc_tape.watch(self.discriminator.trainable_variables)
            # Evaluate on models
            probs = tf.squeeze(self.actor((image_buf, move_buf)))
            reward_cur = tf.squeeze(self.critic(result_buf[0]))
            reward_future = tf.squeeze(self.critic(result_buf[1]))
            disc_cur = tf.squeeze(self.discriminator(result_buf[0]))
            disc_future = tf.squeeze(self.discriminator(result_buf[1]))
            disc_good = tf.squeeze(self.discriminator(good_image))
            advantage = reward_future*GAMMA - reward_cur
            advantage_disc = disc_future*GAMMA - disc_cur
            log_probs = tf.math.log(probs)
            # Mask based on move chosen (OPTIONAL)
                # If the network predicts with 100% probability, other moves will have 0 probability
                # and log (0) is undefined
            # move_chosen = np.reshape(move_chosen, (2,4))
            # log_probs = tf.boolean_mask(log_probs, move_chosen)
            actor_loss = -1 * log_probs * advantage_disc # + 0.5 * log_probs * advantage
            critic_loss = self.actual_future_reward - reward_cur
            disc_loss = disc_cur + (1 - disc_good)
        gradients_of_actor = act_tape.gradient(actor_loss, self.actor.trainable_variables)
        gradients_of_critic = crit_tape.gradient(critic_loss, self.critic.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        if REALTIME_LEARNING:
            self.actor_opt.apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        else:
            self.actor_opt.apply_gradients(zip(gradients_of_actor, self.edit_actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        self.disc_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return actor_loss, critic_loss, disc_loss
