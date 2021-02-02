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
    mn = image.min()
    mx = image.max()
    mx -= mn
    image = ((image-mn)/mx)*255
    return image.astype(np.uint8)


# %%MAIN
if __name__ == '__main__':
    pass
