import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tf.keras.losses import MeanAbsoluteError

checkpoint_path = "actor/actor.ckpt"
checkpoint_path_critic = "critic/critic.ckpt"

# Callback for setting checkpoints
# Will resave weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)

cp_callback_critic = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_critic,
        verbose=1,
        save_weights_only=True,
        period=5)

class Tetris:
    def __init__(self, actor, critic):
        if actor:
            self.actor = __create_model_input()
            print("Successfully created actor")
        if critic:
            self.critic = __create_model_critic()
            print("Successfully created critic")

    def gen_actor(self):
        self.actor = __create_model_input()
        print("Successfully created actor")

    def gen_critic(self):
        self.critic = __create_model_critic()
        print("Successfully created critic")

    def __create_model_input(self):
        model = models.Sequential()
        model.add(layers.Conv2D(120, (10, 10), activation='relu', input_shape=(640,360,3)))
        model.add(layers.MaxPooling2D((5, 5)))
        model.add(layers.Conv2D(240, (10, 10), activation='relu'))
        model.add(layers.MaxPooling2D((5, 5)))
        model.add(layers.Conv2D(240, (10, 10), activation='relu'))


        model.add(layers.Flatten())
        model.add(layers.Dense(240, activation='relu'))
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(10, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        try:
            model.load_weights(checkpoint_path)
        except:
            print('Error loading weights from file')
        return model

    def __create_model_critic(self):
        model = models.Sequential()
        model.add(layers.Conv2D(120, (10, 10), activation='relu', input_shape=(640,360,3)))
        model.add(layers.MaxPooling2D((5, 5)))
        model.add(layers.Conv2D(240, (10, 10), activation='relu'))
        model.add(layers.MaxPooling2D((5, 5)))
        model.add(layers.Conv2D(240, (10, 10), activation='relu'))


        model.add(layers.Flatten())
        model.add(layers.Dense(240, activation='relu'))
        model.add(layers.Dense(2, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        try:
            model.load_weights(checkpoint_path_critic)
        except:
            print('Error loading weights from file')
        return model

    def eval_actor(self, image, buffer):
        pass

    def eval_critic(self, image):
        return self.critic(image)

    def fit_critic(self, image, output):
        self.critic.fit(image, output, callbacks=[cp_callback_critic])
