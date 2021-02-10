'''
DOC
'''
# %% IMPORTS
import numpy as np
# %% FUNCTIONS


def rgb2gray(rgb_image):
    '''
    Receives an ndarray float image and returns the grayscale version.
    '''
    return (rgb_image@[0.2989, 0.5870, 0.1140]).astype(np.float64)


def normalizeuint8(image):
    '''
    Receives float image and converts it to uint8 type.
    '''
    min_val = image.min()
    max_val = image.max()
    max_val -= min_val
    image = ((image-min_val)/max_val)*255
    return image.astype(np.uint8)


# %%MAIN
if __name__ == '__main__':
    pass
