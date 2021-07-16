# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:39:24 2021

@author: yabdulkadir
"""


import tensorflow as tf
import numpy as np
import cv2
from Utils import get_data_len, get_data, train_data_augmentation, val_data_generator, get_raw_data
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

ROIName = 'rectum'

# Dimensions of one slice
img_rows = 300
img_columns = 334
img_channel = 1


with open('{ROIName}_slices_range.txt'.format(ROIName = ROIName), 'r') as femr:
    while True:
        try:
            the_line = femr.readline()
            the_line = the_line[:-1]
            the_split = the_line.split(': ')
            the_key = the_split[0]
            the_value = the_split[1][1:-1]
            the_value_split = the_value.split(', ')
            #print(int(the_value_split[1]))
            #the_diff = int(the_value_split[1]) - int(the_value_split[0])
            #upper_lim = int(the_value_split[1]) + the_diff
            upper_lim = int(the_value_split[1])
            #print("upper_lim: ", upper_lim)
            #if upper_lim > 287:
            #    upper_lim = 287
            the_value = (int(the_value_split[0]), upper_lim)
            #print("the_value: ", the_value)           
            training_set[the_key] = the_value
        except Exception as inst:
            #print(type(inst))
            break
        
        
train_len = get_data_len(training_set)
X_data = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)
Y_data_rectum = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)
Y_data_femr_rt = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)
Y_data_femr_lt = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)


print("Training len: {}".format(train_len))

# Get the training data
X_data, Y_data_rectum, Y_data_femr_rt, Y_data_femr_lt = get_data(training_set, X_data, Y_data_rectum, Y_data_femr_rt, Y_data_femr_lt)

X_raw_data = np.zeros((train_len, img_rows, img_columns, img_channel), dtype = np.float32)
X_raw_data = get_raw_data(training_set, X_raw_data)
Y_body_mask = X_raw_data > 50

def collect_bboxes(input_array, ROI):
    """
    

    Parameters
    ----------
    input_array : TYPE
        DESCRIPTION.
    ROI : TYPE
        DESCRIPTION.

    Returns
    -------
    bboxes : TYPE
        DESCRIPTION.

    """
    bbvalues = []
    for i in range(train_len):
        contours, hierarchy = cv2.findContours(input_array[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            
            if ROI == 'body':
                if len(contours) > 1:
                    contour_indx = 0
                    max_indices = 0
                    for i in range(len(contours)):
                        if contours[i].shape[0] > max_indices:
                            max_indices = contours[i].shape[0]
                            contour_indx = i
                    cnt = contours[contour_indx]
                else:
                    cnt = contours[0]   
            else:
                cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
        except:
            if ROI == 'femr_rt':
                x,y,w,h = (114 - 60), 134, 53, 60
            elif ROI == 'femr_lt':
                x,y,w,h = (204 + 60), 134, 53, 60
            else:
                x,y,w,h = 114, 134, 53, 60
            
        
        bbvalues.append([x,y,w,h])
    bboxes = np.array(bbvalues)
    
    return bboxes


Y_bboxes = collect_bboxes(Y_data_rectum, ROIName)
body_bboxes = collect_bboxes(Y_body_mask, 'body')
femr_rt_bboxes = collect_bboxes(Y_data_femr_rt, 'femr_rt')
femr_lt_bboxes = collect_bboxes(Y_data_femr_lt, 'femr_lt')



y_1_values = (femr_rt_bboxes[:,1] + femr_rt_bboxes[:,3]) * 0 + 0.38
x_1_values = (femr_rt_bboxes[:,0] + femr_rt_bboxes[:,2])/334
y_2_values = (body_bboxes[:,1] + body_bboxes[:,3]) / 300
x_2_values = (femr_lt_bboxes[:,0]) / 334

boxes_ = np.stack([y_1_values, x_1_values, y_2_values, x_2_values], axis = 1)


box_indices_ = np.array(np.arange(X_data.shape[0]))
crop_size_ = np.array([128, 128])
cropped = tf.image.crop_and_resize(X_data, boxes_, box_indices_, crop_size_).numpy()
cropped_masks = tf.image.crop_and_resize(Y_data_rectum, boxes_, box_indices_, crop_size_).numpy()


# Shuffle the training data
train_cases = X_data.shape[0]
ind_list = [i for i in range(train_cases)]
shuffle(ind_list)
cropped = cropped[ind_list]
cropped_masks = cropped_masks[ind_list]

split_length = int(X_data.shape[0]*0.9)

X_train = cropped[:split_length]
Y_train = cropped_masks[:split_length]

X_val = cropped[split_length:]
Y_val = cropped_masks[split_length:]




print("X_train.shape: ", X_train.shape)

# Build the model       
model = UNet_pp_model(4, organ = 'rectum', mode = 'accurate')

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
    
checkpointer = tf.keras.callbacks.ModelCheckpoint(weights_path + "\\unet_pp_{ROIName}_{time_}.h5".format(time_ = time.time(), ROIName = ROIName), verbose = 1, save_best_only = True)

callbacks_ = [tf.keras.callbacks.EarlyStopping(patience = 30, monitor = 'val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path), checkpointer]

model_path = "..\\logs\\Best_Models\\model_{ROIName}_11_16".format(ROIName = ROIName)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Define the batch_size
batch_size = 8

results = model.fit(train_data_augmentation(X_train, Y_train, batch_size), steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size)), batch_size = batch_size, epochs = 150, validation_data = val_data_generator(X_val, Y_val, batch_size), validation_steps =int(np.ceil(X_val.shape[0] / batch_size)),  callbacks = callbacks_)

model.save(model_path)

