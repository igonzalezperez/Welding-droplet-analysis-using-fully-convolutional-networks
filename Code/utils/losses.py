'''
DOCSTRING
'''
import tensorflow as tf


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(
        y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def iou_coef(y_true, y_pred, smooth=1):
    '''
    Doc
    '''
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, [1, 2, 3])+tf.keras.backend.sum(y_pred, [1, 2, 3])-intersection
    iou = tf.keras.backend.mean(1 -
                                (intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef_v1(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) +
                                           tf.keras.backend.sum(y_pred_f) + smooth)


def dice_coef_v2(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(tf.keras.backend.square(y_true), -1)
                                           + tf.keras.backend.sum(tf.keras.backend.square(
                                               y_pred), -1) + smooth)


def dice_loss_v1(y_true, y_pred):
    return 1-dice_coef_v1(y_true, y_pred)


def dice_loss_v2(y_true, y_pred):
    return 1-dice_coef_v2(y_true, y_pred)
