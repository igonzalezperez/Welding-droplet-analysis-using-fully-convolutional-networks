'''
DOC
'''
import cv2
import numpy as np
from PIL import Image
from pycine.color import color_pipeline
from pycine.raw import read_frames


def display_frames(cine_file, start_frame=1, count=1, metadata=False):
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
