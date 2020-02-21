import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import MeanAbsoluteError

tf.keras.backend.set_floatx('float64')

IM_WIDTH_CRITIC = 160
IM_HEIGHT_CRITIC = 90
SHORT_TERM_BUF = 10

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
        self.previous_moves = []
        self.previous_inputs = []
        self.internal_state = tf.zeros(10)
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
        model = models.Sequential()
        model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(IM_HEIGHT_CRITIC,IM_WIDTH_CRITIC,3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(60, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))


        model.add(layers.Flatten())
        model.add(layers.Dense(240, activation='relu'))
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(10, activation='sigmoid'))

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
        model.add(layers.Conv2D(60, (5, 5), activation='relu', input_shape=(IM_HEIGHT_CRITIC,IM_WIDTH_CRITIC,3)))
        model.add(layers.AveragePooling2D((3, 3)))
        model.add(layers.Conv2D(30, (5, 5), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        try:
            model.load_weights(checkpoint_path_critic)
        except:
            print('Error loading weights from file')
        return model

    def eval_actor(self, image, buffer):
        pass

    def eval_critic(self, image):
        return self.critic(image).numpy()

    def fit_critic(self, input_generator, spe, epochs=100):
        self.critic.fit(input_generator, epochs = epochs, verbose=1, steps_per_epoch = spe, callbacks = [cp_callback_critic])
