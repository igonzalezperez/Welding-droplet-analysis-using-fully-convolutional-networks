'''Save .cine video file to .npz array of shape (number_of_frames, height, witdth, rgb_channels).
A grayscale version is also stored which doesn't have the channels dimension.'''
# %% IMPORTS
import os
import numpy as np
from progressbar import progressbar as progress
from pycine.file import read_header
from utils.cine import display_frames
from utils.preprocessing import rgb2gray, normalizeuint8
# %% VARIABLES
DATASET = ('Globular', 'Spray')

# %% FUNCTIONS


def main():
    '''Convert cine file to frames and save them in a .npz file.'''
    for d in DATASET:
        if d == 'Globular':
            video_name = '11-12-06_GMAW CV High V Globular - 175 ipm WFS, 33V, 85-15CO2, 20 ipm travel'
        elif d == 'Spray':
            video_name = '11-12-06_GMAW CV Spray - 400 ipm WFS, 35V, 85-15CO2, 20 ipm travel'

        video_path = os.path.join(
            'Data', 'Video', 'CINE', video_name + '.cine')
        total_frames = read_header(video_path)["cinefileheader"].ImageCount
        save_folder = os.path.join('Data', 'Image', 'Input', d.lower())
        n_frames = total_frames

        data_rgb = []
        data_gray = []
        for i in progress(range(n_frames)):
            # read each frame from .cine file as an ndarray
            rgb_image, _ = display_frames(video_path, start_frame=i+1)
            rgb_image[np.isnan(rgb_image)] = 0
            gray_image = rgb2gray(rgb_image)

            # Convert from float to uint8
            rgb_image = normalizeuint8(rgb_image)
            gray_image = normalizeuint8(gray_image)
            data_rgb.append(rgb_image)
            data_gray.append(gray_image)

        data_rgb = np.array(data_rgb)
        data_gray = np.array(data_gray)

        np.savez_compressed(save_folder + '_rgb_uncompressed', images=data_rgb)
        np.savez_compressed(
            save_folder + '_gray_uncompressed', images=data_gray)


# %% MAIN
if __name__ == "__main__":
    main()
