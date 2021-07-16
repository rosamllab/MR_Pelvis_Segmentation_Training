# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:41:26 2021

@author: yabdulkadir
"""

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator


def mean_zero_normalization(array_):
    """
    Normalizes the array_ by subtracting the mean of the pixel values above a threshold and dividing it by the standard 
    deviation. The values are then truncated to be between -5 and 5 and then normalized to be between 0 and 1.

    Parameters
    ----------
    array_ : numpy array
        Image pixels

    Returns
    -------
    array_ : numpy array
        Normalized image pixels.

    """
    thresholded_pos = array_[array_ > 50]
    mean_ = np.mean(thresholded_pos)
    std_ = np.std(thresholded_pos)
    array_ = (array_ - mean_) / std_
    array_[array_ > 5] = 5
    array_[array_ < -5] = -5
    array_ = array_ + abs(np.min(array_))
    array_ = array_ / np.max(array_)
   
    return array_


def get_data_len(data_set):
    """
    Gets the length of the data.

    Parameters
    ----------
    data_set : dictionary
        Each key represents a patient and the values are the ranges of slices to be extracted for training.

    Returns
    -------
    data_len : int
        The length of the data (number of slices to train).

    """
    parent_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(parent_path, "Data\\Anonymized_Data\\Patients\\{patient_number}")
    data_len = 0

    
    for key in data_set.keys():
        patient_path = data_path.format(patient_number = key)
        the_value = data_set[key]
        pixels_path = os.path.join(patient_path, 'Extracted\\Pixel_arrays')
        patient_pixels = []
        for i in range(the_value[0], the_value[1]):
            patient_pixels.append(np.load(os.path.join(pixels_path, 'Images_{}.npy'.format(i))))
        data_len += len(patient_pixels)
    
    return data_len


def get_data(data_set, X_array, Y_array, ROIName):
    """
    Imports the data for training.

    Parameters
    ----------
    data_set : dictionary
        Each key represents a patient and the values are the ranges of slices to be extracted for training..
    X_array : numpy array
        Zeros array to store the input data.
    Y_array : numpy array
        Zeros array to store the output data.

    Returns
    -------
    X_array : numpy array
        The input data for training.
    Y_array : numpy array
        The output data for training.

    """
    parent_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(parent_path, "Data\\Anonymized_Data\\Patients\\{patient_number}")
    indx = 0
    
    for key in data_set.keys():
        patient_path = data_path.format(patient_number = key)
        the_value = data_set[key]
        pixels_path = os.path.join(patient_path, 'Extracted\\Pixel_arrays')
        masks_path = os.path.join(patient_path, 'Extracted\\Masks_{ROIName}')
        
        for i in range(the_value[0], the_value[1]):
            X_array[indx,:,:,0] = np.load(os.path.join(pixels_path, 'Images_{}.npy'.format(i))).astype(np.float32)
            X_array[indx] = mean_zero_normalization(X_array[indx])
            Y_array[indx,...,0] = np.load(os.path.join(masks_path, 'Masks_{}.npy'.format(i))).astype(np.float32)
            indx += 1

           
    return X_array, Y_array

def get_raw_data(data_set, X_array):
    """
    

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    X_array : TYPE
        DESCRIPTION.

    Returns
    -------
    X_array : TYPE
        DESCRIPTION.

    """
    parent_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(parent_path, "Data\\Anonymized_Data\\Patients\\{patient_number}")
    indx = 0
    
    for key in data_set.keys():
        patient_path = data_path.format(patient_number = key)
        the_value = data_set[key]
        pixels_path = os.path.join(patient_path, 'Extracted\\Pixel_arrays')
        
        for i in range(the_value[0], the_value[1]):
            X_array[indx,:,:,0] = np.load(os.path.join(pixels_path, 'Images_{}.npy'.format(i))).astype(np.float32)            
            indx += 1

           
    return X_array   

    
def train_data_augmentation(imgs, masks, batch_size):
    """
    A generator that augments the training input and output data with the same parameters.

    Parameters
    ----------
    imgs : numpy array
        Input training data.
    masks : numpy array
        Output training data.
    batch_size : int
        The batch size to augment at a time.

    Returns
    -------
    Yields the augmented training input and output data.

    """
    # create two instances with the same arguments
    # create dictionary with the input augmentation values
    data_gen_args = dict(rotation_range = 10, 
                         width_shift_range = 0.1, 
                         height_shift_range = 0.1, 
                         zoom_range = 0.2)
    
    # use this method with both images and masks
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    # fit the augmentation model to the images and masks with the same seed
    image_datagen.fit(imgs, augment = True, seed = seed)
    mask_datagen.fit(masks, augment = True, seed = seed)
    
    # set the parameters for the data to come from images and masks
    image_generator = image_datagen.flow(imgs, 
                                         batch_size = batch_size, 
                                         shuffle = False, 
                                         seed = seed)
    mask_generator = mask_datagen.flow(masks, 
                                       batch_size = batch_size, 
                                       shuffle = False, 
                                       seed = seed)
    
    # combine generators into one which yields image and masks
    # train_generator = zip(image_generator, mask_generator)
    while True:
        yield(image_generator.next(), mask_generator.next())



def val_data_generator(imgs, masks, batch_size):
    """
    A generator that augments the validation input and output data with the same parameters.

    Parameters
    ----------
    imgs : numpy array
        Input training data.
    masks : numpy array
        output training data.
    batch_size : int
        The batch size to augment at a time.

    Returns
    -------
    Yields the augmented validation input and output data.

    """
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    seed = 1
    
    val_img_generator = img_datagen.flow(imgs, 
                                         batch_size = batch_size, 
                                         shuffle = False, 
                                         seed = seed)
    val_mask_generator = mask_datagen.flow(masks, 
                                           batch_size = batch_size, 
                                           shuffle = False, 
                                           seed = seed)
    
    while True:
        yield(val_img_generator.next(), val_mask_generator.next())
        


    
    