from tetris import Tetris
import tensorflow as tf
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 2
BATCH_SIZE = 25
IMG_WIDTH = 160
IMG_HEIGHT = 90

tetris = Tetris(False, True)
data_path = pathlib.Path("training/tetris_critic")
list_train_ds = tf.data.Dataset.list_files(str(data_path/'*/*.png'))
image_count = len(list(data_path.glob("*/*.png")))
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)
CLASS_NAMES = np.array([item.name for item in data_path.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)
CLASS_NAMES = np.array(['neutral', 'negative'])

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float64)
    # resize the image to the desired size.
    return img

def process_path(file_path):
    label = get_label(file_path)
    label = tf.equal(label, True)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

labeled_train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)

def prepare_for_training(ds, cache=False, shuffle_buffer_size=100):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = prepare_for_training(labeled_train_ds)

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==True][0].title())
      plt.axis('off')

image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())
plt.show()

tetris.fit_critic(train_ds, STEPS_PER_EPOCH, epochs=EPOCHS)
