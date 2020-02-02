from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def threshold_binarize(x, threshold=0.85):
    ge = tf.greater_equal(x, tf.constant(threshold))
    return tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))


def iou_thresholded(y_true, y_pred, threshold=0.85, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    return iou_coef(y_true, y_pred, smooth)
