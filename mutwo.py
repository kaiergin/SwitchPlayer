from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Convolution2D, Convolution2DTranspose, Input, Dense, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
import tensorflow as tf
from residual_block import BasicBlock
import numpy as np
import gym
from storage import Storage
import time
import pickle
from cv2 import resize
import threading
import sys

# A tweaked version of MuZero for real-time learning

buffer_size = 16
im_size = 95
expected_dims = (im_size,im_size)
hidden_size = 5
expected_hidden = (hidden_size,hidden_size)
k_paths = 3
future_discount = 0.99
clip_val = .2

output_skip = 400

rep_path = 'model_saves/representation/rep.ckpt'
pol_path = 'model_saves/prediction/pred.ckpt'
dyn_path = 'model_saves/dynamics/dyn.ckpt'

# Helper functions

def one_hot(a, num_classes): return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def k_largest(arr, k):
    found = []
    for _ in range(k):
        max_val = 0
        index = -1
        for i in range(len(arr)):
            if arr[i] > max_val and not i in found:
                max_val = arr[i]
                index = i
        found.append(index)
    return found

def preprocess(frame):
    return resize(frame, expected_dims)

def clip_reward(reward):
    return np.clip(reward, 0, 1)

# Main algorithm, MuTwo
# Uses a learned model to make decisions

class MuTwo:
    def __init__(self, action_space, debug = False):
        self.debug = debug
        self.action_space = action_space
        self.noise_discount = 1.0
        self.depth = 0
        self.clock = 0
        self.training_steps = 0
        self.thread_running = False
        self.representation = self.build_representation()
        self.dynamics = self.build_dynamics()
        self.prediction = self.build_prediction()
        self.representation_old = self.build_representation()
        self.prediction_old = self.build_prediction()
        self.build_action_frames()
        self.opt_dynamics = Adam(1e-4)
        self.opt_prediction = Adam(1e-4)
        if self.debug:
            print(self.rep_frames)
            print(self.dyn_frames)
        self.training_mutex = threading.Lock()
        self.dynamics_mutex = threading.Lock()
        #self.observer = self.build_observer()
        self.storage = Storage()
        self.current_frames = np.zeros((buffer_size, *expected_dims, 3), dtype=np.float32)
        self.current_actions = np.zeros((buffer_size), dtype=np.int32)

    # frame buffer -> hidden state
    # the Network Architecture for MuZero can be found at https://arxiv.org/pdf/1911.08265.pdf on page 14
    def build_representation(self):
        # 3 * buffer_size for images (3 color channels) + buffer_size for actions
        frame = Input(shape=(*expected_dims, 4*buffer_size))
        x = Convolution2D(64, (3,3), strides=2, activation='relu')(frame)
        x = BasicBlock(64)(x)
        x = BasicBlock(64)(x)
        x = Convolution2D(128, (3,3), strides=2, activation='relu')(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = AveragePooling2D(strides=2)(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = AveragePooling2D(strides=2)(x)
        x = sigmoid(x)
        m = Model(inputs=frame, outputs=x)
        if self.debug:
            m.summary()
        try:
            m.load_weights(rep_path)
        except:
            print('No weights to load from')
        return m

    # hidden state + action -> hidden state + reward
    # apparently the dynamics function and representation function have the same architecture but that doesn't make sense because of downscaling?
    def build_dynamics(self):
        # 128 layers are for the hidden state, 129th layer is the action
        hidden_state = Input(shape=(*expected_hidden, 129))
        x = Convolution2D(128, (1,1), activation='relu')(hidden_state)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        x = BasicBlock(128)(x)
        predicted_hidden = sigmoid(x)
        x = Flatten()(x)
        predicited_reward = Dense(1, activation='sigmoid')(x)
        m = Model(inputs=hidden_state, outputs=(predicted_hidden, predicited_reward))
        if self.debug:
            m.summary()
        try:
            m.load_weights(dyn_path)
        except:
            print('No weights to load from')
        return m

    # hidden state -> prediction + value
    def build_prediction(self):
        hidden_state = Input(shape=(*expected_hidden, 128))
        x = BasicBlock(128)(hidden_state)
        # Give each output, value, and error their own basic block?
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        policy = Dense(self.action_space, activation='softmax')(x)
        value = Dense(self.action_space)(x)
        error = Dense(self.action_space)(x)
        m = Model(inputs=hidden_state, outputs=(policy, value, error))
        if self.debug:
            m.summary()
        try:
            m.load_weights(pol_path)
        except:
            print('No weights to load from')
        return m

    # UNFINISHED
    # 16 frame buffer + next frame -> assumed actions
    # To be used as a learning function for watching other players' gameplay
    def build_observer(self):
        frames = Input(shape=(*expected_dims, buffer_size + 1))
        x = Convolution2D(64, (2,2), activation='relu')(frames)
        x = Convolution2D(32, (2,2), activation='relu')(x)
        x = Flatten()(x)
        output = Dense(self.action_space)(x)
        m = Model(inputs=frames, outputs=output)
        if self.debug:
            m.summary()
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    # Creates a tensors of action frames and dynamics frames which can be sliced to get individual frames
    def build_action_frames(self):
        self.rep_frames = []
        self.dyn_frames = []
        for x in range(self.action_space):
            action_rep = np.zeros(expected_dims)
            for y in range(im_size**2):
                if y % self.action_space == x:
                    action_rep[y%im_size][y//im_size] = 1
            self.rep_frames.append(action_rep)
            action_dyn = np.zeros((5,5))
            for y in range(hidden_size**2):
                if y % self.action_space == x:
                    action_dyn[y%hidden_size][y//hidden_size] = 1
            self.dyn_frames.append(action_dyn)
        self.rep_frames = tf.constant(self.rep_frames, dtype=tf.float32)
        self.dyn_frames = tf.constant(self.dyn_frames, dtype=tf.float32)
        assert self.rep_frames.shape == (self.action_space, *expected_dims)
        assert self.dyn_frames.shape == (self.action_space, *expected_hidden)

    # Concatenates an action frame onto a list of frames
    @tf.function
    def concat_action_rep(self, frames, actions):
        frames = tf.concat(tf.unstack(frames, axis=0), -1)
        #assert frames.shape == (*expected_dims, 3*buffer_size)
        #@tf.function
        def get_frame(action):
            frame = tf.slice(self.rep_frames, [action, 0, 0], [1, -1, -1])
            #assert frame.shape == (1, *expected_dims)
            return tf.squeeze(frame)
        action_frames = tf.map_fn(get_frame, actions, dtype=tf.float32)
        # This line might be wrong... not sure if it will effect results
        action_frames = tf.reshape(action_frames, (*expected_dims, buffer_size))
        # possible replacement
        #action_frames = tf.concat(tf.unstack(action_frames, axis=0), -1)
        model_input = tf.concat([frames, action_frames], -1)
        #assert model_input.shape == (*expected_dims, 4*buffer_size)
        return model_input
    
    # Concatenates an action frame onto a hidden state
    @tf.function
    def concat_action_dyn(self, hidden_state, action):
        #assert hidden_state.shape == (*expected_hidden, 128)
        frame = tf.squeeze(tf.slice(self.dyn_frames, [action, 0, 0], [1, -1, -1]))
        action_frame = tf.expand_dims(frame, -1)
        model_input = tf.concat([hidden_state, action_frame], -1)
        #assert model_input.shape == (*expected_hidden, 129)
        return model_input

    # Function for evaluating a single sequence of frames/actions into a hidden state
    @tf.function
    def eval_representation(self, frames, actions, training=False, old=False):
        #print('dtype values:', frames.dtype, actions.dtype)
        model_input = tf.expand_dims(self.concat_action_rep(frames / 255.0, actions), 0)
        if old:
            return tf.squeeze(self.representation_old(model_input))
        return tf.squeeze(self.representation(model_input, training=training))

    # Function for evaluating a single hidden state into a single reward/hidden state
    @tf.function(experimental_relax_shapes=True)
    def eval_dynamics(self, hidden_state, action, training=False):
        model_input = tf.expand_dims(self.concat_action_dyn(hidden_state, action), 0)
        hidden_states, rewards = self.dynamics(model_input, training=training)
        return tf.squeeze(hidden_states), tf.squeeze(rewards)

    # Function for evaluating a single hidden state into a single policy/value/error
    @tf.function
    def eval_prediction(self, hidden_state, training=False, old=False):
        model_input = tf.expand_dims(hidden_state, 0)
        if old:
            policies, values, errors = self.prediction_old(model_input)
        policies, values, errors = self.prediction(model_input, training=training)
        return tf.squeeze(policies), tf.squeeze(values), tf.squeeze(errors)

    # target-value should be actual future discounted reward
    # this function is used for on-policy updates (post game)
    # for regularizing values so that they don't explode from the bootstrapped value function
    @tf.function
    def loss_representation_prediction(self, frames, actions, action_taken, target_value, target_error):
        hidden_state = self.eval_representation(frames, actions, training=True)
        policy, value, error = self.eval_prediction(hidden_state, training=True)
        # Prevents policy value of 0 when taking log
        policy_fixed = tf.nn.softmax(policy + 0.01*tf.ones(self.action_space))
        value_loss = target_value - value
        error_loss = (target_error + value_loss) - error
        policy_loss = -1 * (value_loss + error_loss) * (tf.math.log(policy_fixed) * tf.one_hot(action_taken, self.action_space))
        return policy_loss + value_loss + error_loss

    # Works for arbitrary TD step of n, always bootstrapped from old policy value
    # n should be between 1 and 4, never larger than BUFFER_SIZE
    # this function is used for off-policy updates from the experience replay buffer, and for on-policy updates (in game)
    @tf.function
    def loss_ppo(self, frames, actions, future_frames, actions_taken, rewards_recieved, not_done, error_dyn):
        # Check for bad inputs
        tf.debugging.check_numerics(frames, 'Frames')
        tf.debugging.check_numerics(future_frames, 'Future frames')
        tf.debugging.check_numerics(rewards_recieved, 'Rewards recieved')
        tf.debugging.check_numerics(error_dyn, 'Error dyn')
        # Evaluate on old and new network
        hidden_state = self.eval_representation(frames, actions)
        old_hidden = self.eval_representation(frames, actions, old=True)
        policy, value, error = self.eval_prediction(hidden_state)
        policy_old, _, _ = self.eval_prediction(old_hidden, old=True)
        # Create ratio for new/old policy
        one_hot = tf.one_hot(actions_taken[0], self.action_space)
        div = tf.boolean_mask(policy, one_hot) / (tf.boolean_mask(policy_old, one_hot) + 0.01)
        value = tf.boolean_mask(value, one_hot)
        error = tf.boolean_mask(error, one_hot)
        tf.debugging.check_numerics(div, 'div')
        # Concats future frames/actions onto current buffer, then slices old frames/actions off
        future_frames = tf.concat([frames, future_frames], 0)[tf.shape(future_frames)[0]:][:][:]
        actions_taken = tf.concat([actions, actions_taken], 0)[tf.shape(actions_taken)[0]:]
        # calculate bootstrapped value
        future_hidden = self.eval_representation(future_frames, actions_taken, old=True)
        _, bootstrapped_value, _ = self.eval_prediction(future_hidden, old=True)
        bootstrapped_value = tf.reduce_max(bootstrapped_value)
        # calculate discountead future reward based on bootstrapped value and n sized TD step
        future_rewards = tf.concat([rewards_recieved, [not_done * bootstrapped_value]], 0)
        discounts = tf.math.cumprod(future_discount * tf.ones(tf.shape(future_rewards)[0]))
        bootstrapped_dfr = tf.reduce_sum(future_rewards * discounts)
        # calculate losses using PPO
        clipped = tf.clip_by_value(div, 1.0 - clip_val, 1.0 + clip_val)
        advantage = bootstrapped_dfr - value
        value_loss = tf.abs(advantage)
        without_grad = tf.stop_gradient(advantage)
        policy_loss = -1 * tf.minimum(without_grad * clipped, without_grad * div)
        error_loss = (error_dyn + tf.abs(without_grad)) - error
        tf.debugging.check_numerics(value_loss, 'Value loss')
        tf.debugging.check_numerics(policy_loss, 'Policy loss')
        tf.debugging.check_numerics(error_loss, 'Error loss')
        return value_loss + policy_loss + error_loss

    @tf.function
    def loss_dynamic(self, hidden_state, action, target_state, target_reward):
        pred_hidden, pred_reward = self.eval_dynamics(hidden_state, action)
        return (tf.reduce_sum(tf.abs(pred_hidden - target_state)) / tf.square(hidden_size)) + tf.abs(pred_reward - target_reward)

    '''
    def train_action(self, frames, target_actions):
        targets = np.array([one_hot(x, self.action_space) for x in target_actions])
        accuracy = self.observer.train_on_batch(x = frames, y = targets)[1]
        if self.debug:
            print('Current action guess accuracy:', accuracy)
    '''
    
    #@tf.function
    def prepare_training_dyn(self, frames, actions, future_frame, action_taken):
        #print(frames.dtype, actions.dtype)
        hidden_state = self.eval_representation(frames, actions)
        next_frames = np.roll(frames, -1, 0)
        next_frames[-1] = future_frame
        next_actions = np.roll(actions, -1, 0)
        next_actions[-1] = action_taken
        target_state = self.eval_representation(next_frames, next_actions)
        return hidden_state, target_state

    def train(self, single_experience=True):
        if single_experience:
            start = time.perf_counter()
            frames, _, actions, (future_frames, rewards_recieved, actions_taken, not_done), (game, index) = self.storage.load_random_replay(buffer_size, future_steps=10)
            if future_frames.shape[0] == 0 or actions_taken.shape[0] == 0:
                print('Error finding future frames')
                print('Game', game, 'Index', index)
                return
            # Don't want to backprop representation on this call
            # Roll frames forward one to allow calculation of dynamics loss
            hidden_state, target_state = self.prepare_training_dyn(frames, actions, future_frames[0], actions_taken[0])
            mid = time.perf_counter()
            loss_dynamics = None
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.dynamics.trainable_variables)
                loss_dynamics = self.loss_dynamic(hidden_state, actions_taken[0], target_state, rewards_recieved[0])
                dyn_grad = tape.gradient(loss_dynamics, self.dynamics.trainable_variables)
                self.dynamics_mutex.acquire()
                self.opt_dynamics.apply_gradients(zip(dyn_grad, self.dynamics.trainable_variables))
                self.dynamics_mutex.release()
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.representation.trainable_variables + self.prediction.trainable_variables)
                loss = self.loss_ppo(frames, actions, future_frames, actions_taken, rewards_recieved, not_done, loss_dynamics)
                prediction_grad = tape.gradient(loss, self.representation.trainable_variables + self.prediction.trainable_variables)
                self.opt_prediction.apply_gradients(zip(prediction_grad, self.representation.trainable_variables + self.prediction.trainable_variables))
            end = time.perf_counter()
            self.training_steps += 1
            if self.training_steps % 20 == 0 and self.debug:
                print('Total steps trained', self.training_steps)
                print('Load time:', mid - start)
                print('train time', end - mid)
        else:
            # Parallel calls w/ batch training
            pass

    def train_live(self, single_experience=True):
        if single_experience:
            pass #frames, actions, target_value = self.storage.load_live()
        else:
            # Parallel calls w/ batch training
            pass
    
    def train_loop(self):
        self.stop = False
        self.thread_running = True
        while True:
            self.train()
            if self.stop:
                self.stop = False
                self.thread_running = False
                break

    # To do: make 1 parallel call to evaluate the dynamics function?
    def expand_state(self, state, all_paths=False):
        if all_paths:
            actions_to_explore = range(self.action_space)
        else:
            policy, _, _ = self.eval_prediction(state, training=False)
            rand = np.random.rand(self.action_space)
            rand = softmax(rand)
            rand *= self.noise_discount
            new_policy = softmax(policy + rand)
            # Choose k largest actions
            actions_to_explore = k_largest(new_policy, k_paths)
        next_states = []
        next_rewards = []
        self.dynamics_mutex.acquire()
        for action in actions_to_explore:
            next_s, reward = self.eval_dynamics(state, np.array(action))
            next_states.append(next_s)
            next_rewards.append(reward)
        self.dynamics_mutex.release()
        return actions_to_explore, next_states, next_rewards

    def step(self, obs, reward):
        obs = preprocess(obs)
        reward = clip_reward(reward)
        # Keep track of speed of step function, must make efficient for real-time learning
        start_time = time.perf_counter()
        self.roll_frame(obs)
        # Calculate action (could base on current policy/value prediction, or could expand using dynamics)
        hidden_state = self.eval_representation(self.current_frames, self.current_actions, old=True)
        policy, value, error = self.eval_prediction(hidden_state, old=True)
        if self.clock % output_skip == 0 and self.debug:
            print('Policy:', policy)
            print('Value:', value)
            print('Error', error)
        actions_taken, next_states, next_rewards = self.expand_state(hidden_state, all_paths=True)
        total_values = np.zeros(self.action_space)
        total_errors = np.zeros(self.action_space)
        for action_taken, pred_state, pred_reward in zip(actions_taken, next_states, next_rewards):
            policy, value, error = self.eval_prediction(pred_state, old=True)
            value = np.max(value)
            #prev_states = [pred_state]
            values = [value]
            '''
            for _ in range(self.depth):
                # Look deeper, but only explore k_paths
                states = []
                for x in prev_states:
                    _, deep, r = self.expand_state(x)
                    for state, reward in zip(deep, r):
                        _, v, _ = self.eval_prediction(state)
                        values.append(v + reward)
                    states += deep
                prev_states = states
            '''
            total_values[action_taken] = (max(values) + pred_reward + 0.01)  
            total_errors[action_taken] = ((-1 if np.random.random() > .5 else 1) * (error[action_taken] * np.random.random() + 0.01))
        soft_values = tf.nn.softmax(total_values)
        soft_errors = self.noise_discount * tf.nn.softmax(total_errors)
        action = np.argmax(tf.cast(policy, tf.float64) + soft_values + soft_errors)
        # Save obs and reward and action in storage
        self.storage.save_frame(obs, reward, action)
        self.roll_action(action)
        end_time = time.perf_counter()
        if self.clock % output_skip == 0 and self.debug:
            sys.stdout.flush()
            print('total_values', total_values)
            print('soft_values', soft_values)
            print('policy', policy)
            print('total_errors', total_errors)
            print('soft_errors', soft_errors)
            print('Step time elapsed', end_time - start_time)
        self.clock += 1
        return action
    
    def update_old(self):
        # Set old policy and representation to current policy and representation
        self.stop = True
        while self.stop and self.thread_running:
            time.sleep(0.01)
        self.representation_old.set_weights(self.representation.get_weights())
        self.prediction_old.set_weights(self.prediction.get_weights())
        # Start the training loop again
        threading.Thread(target=self.train_loop, daemon=True).start()

    def roll_frame(self, frame):
        self.current_frames = np.roll(self.current_frames, -1, 0)
        self.current_frames[-1][:][:][:] = frame

    def roll_action(self, action):
        self.current_actions = np.roll(self.current_actions, -1, 0)
        self.current_actions[-1] = action

    def reset(self):
        self.stop = True
        self.pause = False
        self.storage.new_game()
        self.clock = 0
        # Live train on last game
        # self.train_live()
        self.representation.save_weights(rep_path)
        self.prediction.save_weights(pol_path)
        self.dynamics.save_weights(dyn_path)
        if self.noise_discount < .3:
            self.noise_discount = .3
        else:
            self.noise_discount *= .99
        self.current_frames = np.zeros((buffer_size, *expected_dims, 3), dtype=np.float32)
        self.current_actions = np.zeros((buffer_size), dtype=np.int32)
        threading.Thread(target=self.train_loop, daemon=True).start()

# Testing
if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    agent = MuTwo(env.action_space.n, debug = True)
    # Printing trainable variables and their lengths to make sure they are correct
    print('Representation trainable variables')#, agent.representation.trainable_variables)
    print('Length:', len(agent.representation.trainable_variables))
    print('Dynamics trainable variables')#, agent.dynamics.trainable_variables)
    print('Length:', len(agent.dynamics.trainable_variables))
    print('prediction trainable variables')#, agent.prediction.trainable_variables)
    print('Length:', len(agent.prediction.trainable_variables))
    steps = 0
    try:
        steps = pickle.load(open('steps.p','rb'))
    except:
        pass
    while steps < 10000000:
        done = False
        reward = 0
        obs = env.reset()
        #print('Current frames:')
        #print(agent.current_frames)
        #print('Current actions:')
        #print(agent.current_actions)
        # Test model input functions
        #print('concat frames with actions')
        #model_input = agent.concat_action_rep(agent.current_frames, agent.current_actions)
        while not done:
            action = agent.step(obs, reward)
            obs, reward, done, _ = env.step(action)
            steps += 1
            env.render()
            if steps % 2000 == 0:
                print('--- Current step:', steps, '---')
                print('--- Updating new prediction ---')
                print('Action chosen:', action)
                agent.update_old()
            if steps % 100000 == 0:
                print('Upping depth search parameter')
                agent.depth += 1
                if agent.depth > 4:
                    agent.depth = 4
        pickle.dump(steps, open('steps.p', 'wb'))
        agent.reset()