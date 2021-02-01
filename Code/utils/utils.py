'''Useful preprocessing functions to read, save and modify .cine and .jpg files.'''
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycine.color import color_pipeline
from pycine.raw import read_frames


def display_frames(cine_file, start_frame=1, count=1, show=False, metadata=False):
    '''Reads cine file and extracts frames from start_frame to start_frame + count.
       Can plot each frame and show metadata.
    input: cine_file path to read, start_frame determines where to start and count
           determines how many frames are saved. Show plots each frame and
           metadata prints the setup of the respective frame.
    output: Returns a list with rgb images and their setup object from pycine.
    '''
    raw_images, setup, bpp = read_frames(
        cine_file, start_frame=start_frame, count=count)
    rgb_images = (color_pipeline(raw_image, setup=setup, bpp=bpp)
                  for raw_image in raw_images)
    out_img = []
    out_setup = []
    for i, rgb_image in enumerate(rgb_images):
        frame = start_frame + i
        if setup.EnableCrop:
            rgb_image = rgb_image[setup.CropRect.top: setup.CropRect.bottom +
                                  1, setup.CropRect.left: setup.CropRect.right + 1]
        if setup.EnableResample:
            rgb_image = cv2.resize(
                rgb_image, (setup.ResampleWidth, setup.ResampleHeight))
        if show:
            plt.figure()
            plt.imshow(rgb_image)
            plt.show()
        if metadata:
            display_metadata(setup, frame)
        out_img.append(rgb_image)
        out_setup.append(setup)
    if count == 1:
        return rgb_image, setup
    else:
        return out_img, out_setup


def display_metadata(setup, frame_number):
    '''Print metadata of a specific frame.
    input: setup object from pycine package.'''
    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    ccm = cmCalib / cmCalib.sum(axis=1)[:, np.newaxis]
    ccm2 = cmCalib.copy()
    ccm2[0][0] = 1 - ccm2[0][1] - ccm2[0][2]
    ccm2[1][1] = 1 - ccm2[1][0] - ccm2[1][2]
    ccm2[2][2] = 1 - ccm2[2][0] - ccm2[2][1]
    fTone = np.asarray(setup.fTone)
    whitebalance = np.diag(cmCalib)

    print('\nMetadata:\n')
    print('Frame number: ', frame_number)
    print("fFlare: ", setup.fFlare)
    print("WBGain: ", np.asarray(setup.WBGain))
    print("WBView: ", np.asarray(setup.WBView))
    print("fWBTemp: ", setup.fWBTemp)
    print("fWBCc: ", setup.fWBCc)
    print("cmCalib", cmCalib)
    print("whitebalance: ", whitebalance)
    print("ccm: ", ccm)
    print("ccm2", ccm2)
    print("fOffset: ", setup.fOffset)
    print("fGain: ", setup.fGain)
    print("fGainR, fGainG, fGainB: ", setup.fGainR, setup.fGainG, setup.fGainB)
    print("fGamma, fGammaR, fGammaB: ",
          setup.fGamma, setup.fGammaR, setup.fGammaB)
    print("ToneLabel, TonePoints, fTone",
          setup.ToneLabel, setup.TonePoints, fTone)
    print("fPedestalR, fPedestalG, fPedestalB: ",
          setup.fPedestalR, setup.fPedestalG, setup.fPedestalB)
    print("fChroma: ", setup.fChroma)
    print("fHue: ", setup.fHue)


def load_images_paths(folder_path):
    '''Returns list of all .jpg files path inside folder_path.
    input: folder_path to look for images.
    output: list of image paths.'''
    image_paths = sorted([os.path.join(folder_path, file)
                          for file in os.listdir(folder_path) if file.endswith('.jpg')])
    return image_paths


def preprocess(path, resize_shape=None, grayscale=True):
    '''Resize and convert to B&W original .jpg file
    input: path to file, new image shape (in pixels) and wether to convert to grayscale or not
    output: Transformed image'''
    img = cv2.imread(path)
    img = cv2.resize(img, resize_shape) if resize_shape is not None else img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if grayscale else img
    return img


def chunks(input_list, batch_size):
    '''
    Partitions list in chunks of size batch_size. [1,2,3,4, 5] -> [[1,2],[3,4],[5]] That is a list
    partitioned with batch_size = 2.
    '''
    batch_size = max(1, batch_size)
    return (input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size))


def get_concat_h(im1, im2):
    '''
    Horizontaly join two images of same height.
    '''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def read_data(video_id):
    '''Read .npz data intro numpy array.'''
    data = np.load(str(ProjectPaths().npz_dataset_paths[video_id]) + '.npz')
    x_train = data['x_train']
    x_test = data['x_test']
    return x_train, x_test
    # data_train = data['images_train']
    # data_test = data['images_test']
    # return data_train, data_test


def reshaper(shape, div):
    '''
    Receives a tuple shape (x,y) and checks whether x and y are divisible by div.
    It replaces x and y with the closest numebr divisible by div respectively.
    e.g. ((16, 16), 8) -> (16, 16)
         ((16, 17), 8) -> (16, 16)
         ((17, 16), 8) -> (16, 16)
         ((17, 17), 8) -> (16, 16)
    '''
    new_shape = []
    for ss in shape:
        if ss % div == 0:
            new_shape.append(ss)
        else:
            l = [div * int(ss/div), div * (int(ss/div)+1)]
            dl = [ss - l[0], l[1] - ss]

            if dl[0] == dl[1]:
                new_shape.append(l[0])
            else:
                new_shape.append(l[dl.index(min(dl))])
    return new_shape


def main():
    print('FOLDER TREE:\n')
    ProjectPaths().list_files()

    print('\nMEDIA FILES TREE\n')
    ProjectPaths().show_paths()


if __name__ == '__main__':
    main()
