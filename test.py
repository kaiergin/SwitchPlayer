from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Convolution2D, Convolution2DTranspose, Input, Dense, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
import tensorflow as tf

def make_model():
    m_in = Input(shape=(3))
    x = Dense(20)(m_in)
    out1 = Dense(1, activation='sigmoid')(x)
    out2 = Dense(1, activation='sigmoid')(x)
    out3 = Dense(3, activation='softmax')(x)
    m = Model(inputs=m_in, outputs=(out1, out2, out3))
    m.summary()
    return m

mod = make_model()
print('Starting values')
(out1, out2, out3) = mod(tf.ones(shape=(1,3)))
print('Current out1', out1)
print('Current out2', out2)
print('Current out3', out3)
print('---')

def loss(mod_input, target_one, target_two, action, which):
    (out1, out2, out3) = mod(mod_input)
    loss1 = tf.abs(target_one - out1)
    loss2 = tf.abs(target_two - out2)
    out4 = tf.boolean_mask(tf.squeeze(out3), tf.one_hot(action, 3))
    loss3 = loss1 * tf.math.log(out4)
    print('loss1', loss1)
    print('loss2', loss2)
    print('loss3', loss3)
    out4 = tf.boolean_mask(tf.squeeze(out3), tf.one_hot(action, 3))
    loss3 = tf.abs(tf.stop_gradient(loss2) * tf.math.log(out4))
    if which == 0:
        return loss1
    elif which == 1:
        return loss2
    else:
        return loss3
    #return loss1, loss2, loss3

opt1 = Adam(1e-2)
opt2 = Adam(1e-2)
for x in range(3):
    print('action', x)
    for y in range(3):
        with tf.GradientTape() as tape1:
            l = loss(tf.ones(shape=(1,3)), tf.zeros(shape=(1,1)), tf.ones(shape=(1,1)), x, y)
            grad1 = tape1.gradient(l, mod.trainable_variables)
            opt1.apply_gradients(zip(grad1, mod.trainable_variables))
            print('Current loss total', l)
        (out1, out2, out3) = mod(tf.ones(shape=(1,3)))
        print('Current out1', out1)
        print('Current out2', out2)
        print('Current out3', out3)
