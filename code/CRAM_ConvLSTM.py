#-*- coding:utf-8 -*-

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.layers.convolutional_recurrent import ConvLSTM2D

import tensorflow as tf
from keras.layers import Conv2D, DepthwiseConv2D, Dense, ReLU
from keras.layers import Lambda, Activation
from keras.layers import add, multiply
from keras.backend import var

def CRAM_ConvLSTM(input,weight,height,channels,pretrained_weights=None,kernel_size=3):
        inputs = input
        x1_channels = 32
        ker = 3
	conv1 = Conv3D(x1_channels*4, ker, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        ram1  = CRAM3D(conv1,channels=x1_channels*4, CRAM_number=2)
	cLSM1 = ConvLSTM2D(x1_channels*4, kernel_size=(ker, ker), padding='same', return_sequences=True)(ram1)
        BN1   = BatchNormalization()(cLSM1)

        ram2  = CRAM3D(BN1,channels=x1_channels*4, CRAM_number=2)
	cLSM2 = ConvLSTM2D(x1_channels*4, kernel_size=(ker, ker), padding='same', return_sequences=True)(ram2)
        BN2   = BatchNormalization()(cLSM2)

        ram3  = CRAM3D(BN2,channels=x1_channels*2, CRAM_number=2)
	cLSM3 = ConvLSTM2D(x1_channels*2, kernel_size=(ker, ker), padding='same', return_sequences=True)(ram3)
        BN3   = BatchNormalization()(cLSM3)

        ram4  = CRAM3D(BN3,channels=x1_channels*1, CRAM_number=2)
	cLSM4 = ConvLSTM2D(x1_channels*1, kernel_size=(ker, ker), padding='same', return_sequences=True)(ram4)
        BN4   = BatchNormalization()(cLSM4)
        conv10 = Conv3D(filters=1, kernel_size=(ker, ker, ker), activation='linear', padding='same', data_format='channels_last')(BN4)
	return conv10

def upsampler3D(input, scale, channels=128, kernel_size=3):
    x = Conv3D(channels * (scale ** 2), kernel_size, strides=1, padding='same')(input)
    # x = SubpixelConv3D(x.shape, scale=scale)(x)
    return x

def CRAM3D(input,channels, CRAM_number,kernel_size=3):
        x1_channels = channels
        x1 = x0 = Conv3D(x1_channels, kernel_size, strides=1, padding='same')(input)
        x1 = Conv3D(x1_channels, kernel_size, strides=1, padding='same')(x1)
        x1 = ReLU()(x1)
        x1 = Conv3D(x1_channels, kernel_size, strides=1, padding='same')(x1)

        # compute attentions
        ca = CA3D(x1, x1_channels, CRAM_number)
        sa = SA3D(x1)
        fa = add([ca, sa])
        fa = Activation('sigmoid')(fa)

        # apply attention
        x3 = multiply([x1, fa])
        x3 = add([x0, x3])

        x4 = Conv3D(x1_channels, kernel_size=kernel_size, strides=1, padding='same')(x3)
        x4 = add([x4, x0])
        x4 = upsampler3D(x4, scale=1)
        x4 = Conv3D(x1_channels, kernel_size=kernel_size, strides=1, padding='same')(x4)
	
	return x4

def CA3D(input, channels, reduction_ratio):
    x = Lambda(lambda x: var(x, axis=4, keepdims=True))(input)
    x = Dense(channels // reduction_ratio)(x)
    x = ReLU()(x)
    x = Dense(channels)(x)

    return x

def SA3D(input, kernel_size=3):
    x = TimeDistributed(DepthwiseConv2D(kernel_size, padding='same'))(input)
    return x



