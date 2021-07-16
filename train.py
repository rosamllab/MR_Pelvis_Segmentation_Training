# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:47:07 2021

@author: yabdulkadir
"""

import tensorflow as tf
import numpy as np
from Utils import get_data_len, get_data, train_data_augmentation, val_data_generator
from random import shuffle
from UNetPlusPlus import UNet_pp_model
from metrics_functions import dice_coef, dice_coef_loss
from datetime import date
import os
import time
#import matplotlib.pyplot as plt

# Getting the training data

# A dictionary for the slice ranges of each patients
training_set = {}

ROIName = 'Femr_rt'

# Extract the slice ranges to import as tuples for each patient
with open('{ROIName}_slices_range.txt'.format(ROIName = ROIName), 'r') as roi:
    while True:
        try:
            the_line = roi.readline()
            the_line = the_line[:-1]
            the_split = the_line.split(': ')
            the_key = the_split[0]
            the_value = the_split[1][1:-1]
            the_value_split = the_value.split(', ')
            #print(int(the_value_split[1]))
            the_diff = int(the_value_split[1]) - int(the_value_split[0])
            upper_lim = int(the_value_split[1]) + the_diff
            #print("upper_lim: ", upper_lim)
            if upper_lim > 287:
                upper_lim = 287
            the_value = (int(the_value_split[0]), upper_lim)
            #print("the_value: ", the_value)           
            training_set[the_key] = the_value
        except Exception as inst:
            break
        
        
# Dimensions for one slice
img_rows = 300
img_columns = 334
img_channel = 1

    
train_len = get_data_len(training_set)
X_data = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)
Y_data = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)

print("Training len: {}".format(train_len))

    
# Get the training data
X_data, Y_data = get_data(training_set, X_data, Y_data, ROIName)

# Shuffle the training data
train_cases = X_data.shape[0]
ind_list = [i for i in range(train_cases)]
shuffle(ind_list)
X_data = X_data[ind_list]
Y_data = Y_data[ind_list]

split_length = int(X_data.shape[0]*0.9)

X_train = X_data[:split_length]
Y_train = Y_data[:split_length]

X_val = X_data[split_length:]
Y_val = Y_data[split_length:]


print("X_train.shape: ", X_train.shape)

# Build the model       
model = UNet_pp_model(4, mode = 'accurate')
 
model.summary()        

model.compile(optimizer="Adam", 
              loss = dice_coef_loss, 
              metrics=[dice_coef])

model_path = "..\\logs\\{direc}\\model_{ROIName}_unetpp".format(direc = date.today(), ROIName = ROIName)
print(model_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(model_path)


weights_path = "..\\logs\\{direc}\\weights_{ROIName}_unetpp".format(direc = date.today(), ROIName = ROIName)
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    
tensorboard_path = "..\\logs\\{direc}\\unet_pp_{ROIName}_log_{tpe}".format(direc = date.today(), tpe = time.time(), ROIName = ROIName)
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
    
checkpointer = tf.keras.callbacks.ModelCheckpoint(weights_path + "\\unet_pp_{ROIName}_{t}.h5".format(t = time.time(), ROIName = ROIName), verbose = 1, save_best_only = True)

callbacks_ = [tf.keras.callbacks.EarlyStopping(patience = 30, monitor = 'val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path), checkpointer]



# Define the batch_size
batch_size = 8

results = model.fit(train_data_augmentation(X_train, Y_train, batch_size), 
                    steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size)), 
                    batch_size = batch_size, 
                    epochs = 150, 
                    validation_data = val_data_generator(X_val, Y_val, batch_size), 
                    validation_steps =int(np.ceil(X_val.shape[0] / batch_size)),  
                    callbacks = callbacks_)


# # Prediction on the training data
# preds_train = model.predict(X_train, verbose = 1)
# preds_val = model.predict(X_val, verbose = 1)



# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)


# def plot_slice_with_contour(indx, categ):
#     if categ == "train":
#         plt.imshow(X_train[indx], cmap = 'gray')
#         plt.contour(preds_train_t[indx].reshape(img_rows, img_columns), colors = 'b')
#         plt.axis('off')
#         plt.title('used in training')
#     elif categ == "val":
#         plt.imshow(X_val[indx], cmap = 'gray')
#         plt.contour(preds_val_t[indx].reshape(img_rows, img_columns), colors = 'b')
#         plt.axis('off')
#         plt.title('used in validation')
        
# plot_slice_with_contour(50, "train")

