'''Save each frame of a .cine file to .jpg'''
# %% Imports
import os
import matplotlib.pyplot as plt
from PIL import Image
import progressbar
import numpy as np

from pycine.file import read_header
from utils.utils import display_frames
# %% VARIABLES
DATASET = ('Globular', 'Spray')

# %% FUNCTIONS


def rgb2gray(rgb_image):
    return (rgb_image@[0.2989, 0.5870, 0.1140]).astype(np.float64)


def normalizeuint8(image):
    mn = image.min()
    mx = image.max()
    mx -= mn
    image = ((image-mn)/mx)*255
    return image.astype(np.uint8)


def main():
    '''Convert cine file to frames and save them in an npz file.'''
    for d in DATASET:
        if d == 'Globular':
            video_name = '11-12-06_GMAW CV High V Globular - 175 ipm WFS, 33V, 85-15CO2, 20 ipm travel'
        elif d == 'Spray':
            video_name = '11-12-06_GMAW CV Spray - 400 ipm WFS, 35V, 85-15CO2, 20 ipm travel'

        video_path = os.path.join(
            'Data', 'Video', 'CINE', video_name + '.cine')
        total_frames = read_header(video_path)["cinefileheader"].ImageCount
        save_folder = os.path.join('Data', 'Image', d.lower())
        n_frames = total_frames
        data = []
        data_gray = []
        for i in progressbar.progressbar(range(n_frames)):
            rgb_image, _ = display_frames(video_path, start_frame=i+1)
            rgb_image[np.isnan(rgb_image)] = 0
            gray_image = rgb2gray(rgb_image)

            rgb_image = normalizeuint8(rgb_image)
            gray_image = normalizeuint8(gray_image)
            data.append(rgb_image)
            data_gray.append(gray_image)

        data = np.array(data)
        data_gray = np.array(data_gray)

        np.savez_compressed(save_folder + '_rgb', images=data_gray)
        np.savez_compressed(save_folder + '_gray', images=data_gray)


# %%
if __name__ == "__main__":
    main()
