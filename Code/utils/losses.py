'''
Segmentation loss functions.
'''
# %% IMPORTS
import tensorflow as tf

# %% FUNCTIONS


# def jaccard_distance(y_true, y_pred, smooth=100):
#     """ Calculates mean of Jaccard distance as a loss function """
#     intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
#     sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     jd =  (1 - jac) * smooth
#     return tf.reduce_mean(jd)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    '''
    Alternative implementation for jaccard index.
    '''
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, [1, 2, 3])+tf.keras.backend.sum(y_pred, [1, 2, 3])-intersection
    jaccard_distance = tf.keras.backend.mean(1 -
                                             (intersection + smooth) / (union + smooth), axis=0)
    return jaccard_distance


def dice_coef_v1(y_true, y_pred, smooth=1):
    '''
    Alternative implementation for DICE coefficient.
    '''
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
    '''
    Compute dice loss from dice coefficient.
    '''
    return 1-dice_coef_v1(y_true, y_pred)


def dice_loss_v2(y_true, y_pred):
    '''
    Compute dice loss from dice coefficient.
    '''
    return 1-dice_coef_v2(y_true, y_pred)

# %% MAIN


if __name__ == '__main__':
    pass
