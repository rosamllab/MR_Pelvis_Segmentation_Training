# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:15:58 2020

@author: yabdulkadir
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, ZeroPadding2D, Cropping2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from datetime import date
from metrics_functions import dice_coef, dice_coef_loss
from tqdm import tqdm
import time
from random import shuffle
#from keras.models import load_model, load_weights



# Convolution layer configurations
dropout_rate = 0.2
filters = (32, 64, 128, 256, 512)
kernel_initializer = 'he_normal'
activation = 'relu'
padding = 'same'
kernel_size = (3,3)
strides = (2,2)


# A single encoder block
def encoder_block(stage, 
                  column, 
                  kernel_size, 
                  filters, 
                  strides, 
                  padding, 
                  kernel_initializer, 
                  activation, 
                  blocks, 
                  input_tensor):
    """
    Defines one encoder block of the UNetPlusPlus.

    Parameters
    ----------
    stage : int
        Stage of the encoder block to build.
    column : int
        Column of the encoder block to build.
    kernel_size : tuple
        Kernel size to use in the convolution layers.
    filters : int
        Number of filters to use in the convolution layers.
    strides : tuple
        Strides to use in the convolution layers.
    padding : str
        Padding type to use in the convolution layers.
    kernel_initializer : str
        Kernel initializer to use in the convolution layers.
    activation : str
        The activation function to use in the convolution layers.

    Returns
    -------
    None.

    """
    layer_name = 'x' + str(stage) + str(column)
    if stage == 0:
        inputs = input_tensor
    else:
        inputs = blocks['x' + str(stage - 1) + str(column)]
        inputs = MaxPooling2D((2,2), strides = (2,2))(inputs)
    
    conv_layer = Conv2D(filters, 
                        kernel_size, 
                        activation = activation, 
                        kernel_initializer = kernel_initializer, 
                        padding = padding, 
                        name = 'conv_' + layer_name,
                        kernel_regularizer = l2(1e-04))(inputs)
    dropout_layer = Dropout(dropout_rate, name = 'dropout_' + layer_name)(conv_layer)
    conv_layer = Conv2D(filters, 
                        kernel_size, 
                        activation = activation,
                        kernel_initializer = kernel_initializer, 
                        padding = padding, 
                        name = layer_name,
                        kernel_regularizer = l2(1e-04))(dropout_layer)
    
    blocks['x' + str(stage) + str(column)] = conv_layer
    print('encoder built x' + str(stage) + str(column) + ', shape = ' + str(conv_layer.shape))


# A single decoder block
def decoder_block(stage, 
                  column, 
                  kernel_size, 
                  filters, 
                  padding, 
                  activation, 
                  strides, 
                  kernel_initializer, 
                  blocks):
    """
    Defines one decoder block of the UNetPlusPlus.

    Parameters
    ----------
    stage : int
        Stage of the decoder block to build.
    column : int
        Column of the decoder block to build.
    kernel_size : tuple
        Kernel size to use in the convolution layers.
    filters : int
        Number of filters to use in the convolution layers.
    padding : str
        Padding type to use in the convolution layers.
    activation : str
        The activation function to use in the convolution layers.
    strides : tuple
        Strides to use in the convolution layers.
    kernel_initializer : str
        Kernel initializer to use in the convolution layers.

    Returns
    -------
    None.

    """
    layer_name = 'x' + str(stage) + str(column)
    if column > 0:
        inputs = blocks['x' + str(stage + 1) + str(column - 1)]
        transpose_layer = Conv2DTranspose(filters, 
                                          kernel_size, 
                                          strides, 
                                          padding, 
                                          name = 'upconv_' + layer_name)(inputs)
        to_conc = [transpose_layer]
        for i in range(column):
            to_conc.append(blocks['x' + str(stage) + str(i)])
        conc_layer = concatenate(to_conc, name = 'conc_' + layer_name)
    
        conv_layer = Conv2D(filters, 
                            kernel_size, 
                            activation = activation,
                            kernel_initializer = kernel_initializer, 
                            padding = padding, 
                            name = 'conv_' + layer_name,
                            kernel_regularizer = l2(1e-04))(conc_layer)
        dropout_layer = Dropout(dropout_rate, name = 'dropout_' + layer_name)(conv_layer)
        conv_layer = Conv2D(filters, 
                            kernel_size, 
                            activation = activation,
                            kernel_initializer = kernel_initializer, 
                            padding = padding, 
                            name = layer_name,
                            kernel_regularizer = l2(1e-04))(dropout_layer)
        blocks['x' + str(stage) + str(column)] = conv_layer
        print('decoder built x' + str(stage) + str(column) + ', shape = ' + str(conv_layer.shape))
    

    
def build_encoders(layers, filters, blocks, input_tensor):
    """
    Builds all the encoder blocks of the UNetPlusPlus with the input number of layers.

    Parameters
    ----------
    layers : int
        Number of layers (stages) of the UNetPlusPlus model.
    filters : tuple
        Number of filters for each layer of the model.

    Returns
    -------
    None.

    """
    print("Building encoders...")
    for s in tqdm(range(layers + 1)):
        encoder_block(s, 
                      0, 
                      kernel_size, 
                      filters[s], 
                      strides, 
                      padding, 
                      kernel_initializer, 
                      activation, 
                      blocks,
                      input_tensor)


def build_decoders(layers, filters, blocks):
    """
    Builds the decoder blocks of the UNetPlusPlus with the input number of layers.

    Parameters
    ----------
    layers : int
        Number of layers (stages) of the UNetPlusPlus model.
    filters : tuple
        Number of filters for each layer of the model.

    Returns
    -------
    None.

    """
    print("Building decoders...")
    if layers == 1:
        decoder_block(0,
                      1, 
                      kernel_size, 
                      filters[0], 
                      padding, 
                      activation, 
                      strides, 
                      kernel_initializer,
                      blocks)
    else:
        for i in tqdm(range(layers)):
            s = layers - i - 1
            for j in range(i + 1):
                c = j + 1
                decoder_block(s,
                              c, 
                              kernel_size, 
                              filters[s], 
                              padding, 
                              activation, 
                              strides, 
                              kernel_initializer,
                              blocks)

    

def UNet_pp_model(layers, organ = None, **kwargs):
    """
    Builds the entire UNetPlusPlus model with the input number of layers.

    Parameters
    ----------
    input_tensor : Tensor
        The input tensor (first tensor) of the model.
    layers : int
        Number of layers of the UNetPlusPlus model.
    **kwargs : key, word pairs to define the mode of the model.
        If mode is set to fast, it uses deep supervision and if set to accurate, it does not.

    Returns
    -------
    model : class
        The UNetPlusPlus model.

    """
    
    # Initialize a dictionary to store the blocks
    blocks = {}
    
    if organ == 'rectum':
        
        # Define dimensions
        img_rows = 128
        img_columns = 128
        img_channel = 1
        
        # Define the input tensor
        image_tensor = Input((img_rows, img_columns, img_channel))
        input_tensor = image_tensor
    else:
        
        # Define dimensions
        img_rows = 300
        img_columns = 334
        img_channel = 1
        
        # Define the input tensor
        image_tensor = Input((img_rows, img_columns, img_channel))
        input_tensor = Cropping2D(cropping=((6,6), (7,7)))(image_tensor)
    
    
    build_encoders(layers, filters, blocks, input_tensor)
    print(' ')
    build_decoders(layers, filters, blocks)
    mode = None
    fast_layer = None
    for key, value in kwargs.items():
        if key == 'mode' and value == 'accurate':
            mode = 'accurate'
        elif key == 'mode' and value == 'fast':
            mode = 'fast'
        elif key == 'layer':
            fast_layer = int(value)
    if mode == 'accurate':
        if layers == 1:
            outputs = blocks['x' + str(0) + str(1)]
            outputs = Conv2D(1, 
                             (1,1), 
                             activation = 'sigmoid', 
                             kernel_initializer = kernel_initializer, 
                             padding = 'same',
                             kernel_regularizer = l2(1e-04))(outputs)
            outputs = ZeroPadding2D(padding = ((6,6), (7,7)))(outputs)
        else:
            outputs = blocks['x' + str(0) + str(layers)]
            outputs = Conv2D(1, 
                             (1,1), 
                             activation = 'sigmoid',
                             kernel_initializer = kernel_initializer, 
                             padding = 'same',
                             kernel_regularizer = l2(1e-04))(outputs)
            outputs = ZeroPadding2D(padding = ((6,6), (7,7)))(outputs)
        
        model = Model(inputs = image_tensor, outputs = outputs)
    elif mode == 'fast':
        out_list = ['x' + str(0) + str(c + 1) for c in range(fast_layer)]
        outputs = [Conv2D(1, 
                          (1, 1), 
                          activation = 'sigmoid',
                          kernel_initializer = kernel_initializer,
                          padding = 'same',
                          kernel_regularizer = l2(1e-04))(output) for output in out_list]
        
        model = Model(inputs = image_tensor, outputs = outputs)
    
    else:
        raise ValueError("mode must be set to either accurate or fast")
    
    
    return model

#########################################################################################################

if __name__ == '__main__':

    # Build the model       
    model = UNet_pp_model(4, mode = 'accurate')
     
    model.summary()        

    


