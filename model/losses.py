import keras.backend as K
from keras.losses import binary_crossentropy
from .metrics import dice_coef, iou_coef

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_coef_loss(y_true, y_pred):
    return 1.-iou_coef(y_true, y_pred)