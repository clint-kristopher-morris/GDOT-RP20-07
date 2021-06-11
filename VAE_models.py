import os, sys, time
import numpy as np
import sympy
from sympy import *
import pandas as pd

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
#import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import scaleogram as scg

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.backend import flatten
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Reshape, Conv2DTranspose, Layer, Lambda, MaxPooling2D
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

"""
CCS data was previously passed through the VAE
the result is loaded from a csv file
"""
df_lspace = pd.read_csv('data/latentspaceV2.csv')
df_lspace = df_lspace[[str(x) for x in range(64)]]
space = df_lspace.to_numpy()


"""
VAE Parameters
"""
reshape_size_recur = 32 # move later
n_scales = 64
reshape_size = 64
interval_len = 288
channel_count = 1
latent_dim = 32
"""
Load the VAE for the recurrence plot
"""
shape_after_encoder = (8, 8, 1024)
name = 'model_recurr'
## encoder model
input_encoder = Input(shape=(reshape_size_recur,reshape_size_recur,channel_count), name='input_encoder')
x = Conv2D(32, 3, padding='same', activation='relu', name='conv_00')(input_encoder)
x = Conv2D(32, 3, padding='same', activation='relu', name='conv_1')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Conv2D(64, 3, padding='same', activation='relu', name='conv_6')(x)
x = Conv2D(64, 3, padding='same', activation='relu', name='conv_7')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Conv2D(128, 3, padding='same', activation='relu', name='conv_87')(x)
x = Conv2D(128, 3, padding='same', activation='relu', name='conv_77')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Flatten(name='flatten_0')(x)
# x = Dropout(0.2, name='dropout')(x)
z_mean = Dense(latent_dim, name='dense_z_mean')(x)
z_log_var = Dense(latent_dim, name='dense_z_log_var')(x)
encoder_model_recur = Model(input_encoder, [z_mean, z_log_var], name='encoder_model_recur')

## sampling layer
def sampling_recur(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + tf.math.exp(z_log_var)*epsilon
sampling_layer_recur = Lambda(sampling_recur, name='sample')


## decoder model
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')
x = Dense(tf.math.reduce_prod(shape_after_encoder), activation='relu', name='dense_1')(decoder_input)
x = Reshape(shape_after_encoder, name='reshape_0')(x)
x = Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2,2), name='cond_2d_transpose_26')(x)
x = Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2,2), name='cond_2d_transpose_52')(x)
x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(1,1), name='cond_2d_transpose_2')(x)
x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(1,1), name='cond_2d_transpose_4')(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(1,1), name='cond_2d_transpose_3')(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(1,1), name='cond_2d_transpose_7')(x)
z_decoded = Conv2D(channel_count, 2, padding='same', activation='sigmoid', name='conv_4')(x)
decoder_model_recur = Model(decoder_input, z_decoded, name='decoder_model_recur')

## losses layer
class CustomVariationalLayer(Layer):
    def vae_loss(self, x, z_decoded):
        x = flatten(x)
        z_decoded =  flatten(z_decoded)
        xent_loss = binary_crossentropy(x, z_decoded)
        kl_loss_weight = -5e-4
        kl_loss = tf.math.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.math.exp(self.z_log_var), axis=-1)
        return tf.reduce_mean(xent_loss + kl_loss_weight*kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        self.z_mean = inputs[2]
        self.z_log_var = inputs[3]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
losses_layer = CustomVariationalLayer(name='custom_loss')

## VAE model definition
input_img = Input(shape=(reshape_size_recur,reshape_size_recur,channel_count), name='input_image')
z_mean, z_log_var = encoder_model_recur(input_img)
z = sampling_layer_recur([z_mean, z_log_var])
z_decoded = decoder_model_recur(z)

y_recur = losses_layer([input_img, z_decoded, z_mean, z_log_var])

vae_recur = Model(input_img, y_recur)
vae_recur.compile(optimizer='rmsprop', loss=None)

latest = tf.train.latest_checkpoint('models/vae_r7')
# Load the previously saved weights
vae_recur.load_weights(latest)








"""
Load the VAE for the wavelet plot
"""
name = 'model_5'
channel_count = 1
latent_dim = 32
shape_after_encoder = (8, 8, 1024)

## encoder model
input_encoder = Input(shape=(reshape_size,reshape_size,channel_count), name='input_encoder')
x = BatchNormalization()(input_encoder)
x = Conv2D(64, 3, padding='same', activation='relu', name='conv_0')(x)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(1, 1), name='conv_1')(x)
x = Conv2D(64, 3, padding='same', activation='relu', name='conv_6')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Conv2D(128, 3, padding='same', activation='relu', name='conv_66')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Flatten(name='flatten_0')(x)

# x = Dense(1024, activation='relu', name='dense_2')(x)
# x = Dropout(0.2, name='dropout')(x)
z_mean = Dense(latent_dim, name='dense_z_mean')(x)
z_log_var = Dense(latent_dim, name='dense_z_log_var')(x)
encoder_model = Model(input_encoder, [z_mean, z_log_var], name='encoder_model')

## sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + tf.math.exp(z_log_var)*epsilon

sampling_layer = Lambda(sampling, name='sample')

## decoder model
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')
x = Dense(tf.math.reduce_prod(shape_after_encoder), activation='relu', name='dense_1')(decoder_input)
x = Reshape(shape_after_encoder, name='reshape_0')(x)
x = Conv2DTranspose(128, 3, padding='same', activation='relu', strides=(2,2), name='cond_2d_transpose_00')(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2), name='cond_2d_transpose_1')(x)
x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2), name='cond_2d_transpose_3')(x)
# x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(1,1), name='cond_2d_transpose_4')(x)
z_decoded = Conv2D(channel_count, channel_count, padding='same', activation='sigmoid', name='conv_4')(x)
decoder_model = Model(decoder_input, z_decoded, name='decoder_model')

## losses layer
class CustomVariationalLayer(Layer):
    def vae_loss(self, x, z_decoded):
        x = flatten(x)
        z_decoded =  flatten(z_decoded)
        xent_loss = binary_crossentropy(x, z_decoded)
        kl_loss_weight = -5e-4
        kl_loss = tf.math.reduce_mean(1 + self.z_log_var - tf.square(self.z_mean) - tf.math.exp(self.z_log_var), axis=-1)
        return tf.reduce_mean(xent_loss + kl_loss_weight*kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        self.z_mean = inputs[2]
        self.z_log_var = inputs[3]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

losses_layer = CustomVariationalLayer(name='custom_loss')

## VAE model definition
input_img = Input(shape=(reshape_size,reshape_size,channel_count), name='input_image')
z_mean, z_log_var = encoder_model(input_img)
z = sampling_layer([z_mean, z_log_var])
z_decoded = decoder_model(z)

y = losses_layer([input_img, z_decoded, z_mean, z_log_var])

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)

losses_layer = CustomVariationalLayer(name='custom_loss')

## VAE model definition
input_img = Input(shape=(reshape_size,reshape_size,channel_count), name='input_image')
z_mean, z_log_var = encoder_model(input_img)
z = sampling_layer([z_mean, z_log_var])
z_decoded = decoder_model(z)

y = losses_layer([input_img, z_decoded, z_mean, z_log_var])

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)


latest = tf.train.latest_checkpoint('models/wavelet_r7')
# Load the previously saved weights
vae.load_weights(latest)
