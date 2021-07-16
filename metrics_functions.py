# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 00:11:57 2020

@author: YAbdulkadir
"""

import keras
from keras import backend as K


def dice_coef(y_true, y_pred):
    """
    Computes the soft dice similarity coefficient (DSC).

    Parameters
    ----------
    y_true : Tensor
        The ground truth segmentation mask.
    y_pred : Tensor
        The predicted segmentation mask.

    Returns
    -------
    float
        The soft DSC result.

    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Computes the soft dice loss.

    Parameters
    ----------
    y_true : Tensor
        The ground truth segmentation mask.
    y_pred : Tensor
        The predicted segmentation mask.

    Returns
    -------
    float
        The soft dice loss result.

    """
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """
    Computes the hybrid binary-cross-entropy and soft dice loss.

    Parameters
    ----------
    y_true : Tensor
        The ground truth segmentation mask.
    y_pred : Tensor
        The predicted segmentation mask.

    Returns
    -------
    float
        The result of the hybrid loss.

    """
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)