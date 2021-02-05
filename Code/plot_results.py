'''
DOC
'''
# %% IMPORTS
import os
import sys
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from cv2 import cv2
from progressbar import progressbar as progress
import matplotlib as mpl

mpl.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
# %% VAIRABLES
'''
Pixel to distance conversions
26 px = .045 in = 1.143 mm
1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
'''

ARCHITECTURE_NAME = 'unet'
DATASET = 'Spray'
N_FILTERS = 16
BATCH_SIZE_TRAIN = 16
EPOCHS = 100
GEOMETRY_DIR = os.path.join('Output', 'Geometry',
                            f'{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle')
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')
P1 = 334*10 ** (-6)  # period between frames in [s]
P2 = 333*10 ** (-6)
PX_TO_MM = 4.396153846153846 * 10**(-2)
with open(GEOMETRY_DIR, 'rb') as f:
    GEOMETRY = pickle.load(f)

# %% FUNCTIONS


def parse_centroid_coords(cent):
    x = []
    y = []
    for c in cent:
        x.append(c[0])
        y.append(c[1])
    return x, y


def centroid_img_generator():
    geom = GEOMETRY['centroids']
    data = np.load(os.path.join('Data', 'Image', 'Input',
                                f'{DATASET.lower()}_rgb.npz'))
    images = data['images']
    for cent, image in zip(geom, images):
        yield cent, image


def centroid_mask_img_generator():
    geom = GEOMETRY['centroids']
    data = np.load(os.path.join('Data', 'Image', 'Input',
                                f'{DATASET.lower()}_rgb.npz'))
    data_mask = np.load(PREDS_DIR)
    images = data['images']
    masks = data_mask['preds']

    for cent, image, mask in zip(geom, images, masks):
        yield cent, image, mask


def plot_centroids(save=False):
    frames = len(GEOMETRY['centroids'])-1
    data = centroid_img_generator()
    cent, img = next(data)
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(img)
    x, y = parse_centroid_coords(cent)
    cent_plt, = ax.plot(x, y, 'rx')

    def animate(i):
        try:
            cent, img = next(data)
            im.set_data(img)
            x, y = parse_centroid_coords(cent)
            cent_plt.set_data(x, y)
        except StopIteration as e:
            print(e)

        return im, cent_plt,
    ani = FuncAnimation(fig, animate, interval=90, save_count=frames)
    if save:
        ani.save(f'centroid_{DATASET}.mp4')
    else:
        plt.show()


def plot_centroids_with_masks(save=False):
    frames = len(GEOMETRY['centroids'])-1
    data = centroid_mask_img_generator()
    cent, img, msk = next(data)
    fig, ax = plt.subplots(1, 2)
    im = ax[0].imshow(img)
    mk = ax[1].imshow(msk)
    x, y = parse_centroid_coords(cent)
    cent_plt_1, = ax[0].plot(x, y, 'rx')
    cent_plt_2, = ax[1].plot(x, y, 'rx')

    def animate(i):
        try:
            cent, img, msk = next(data)
            im.set_data(img)
            mk.set_data(msk)
            x, y = parse_centroid_coords(cent)
            cent_plt_1.set_data(x, y)
            cent_plt_2.set_data(x, y)
            return im, mk, cent_plt_1, cent_plt_2
        except StopIteration as e:
            print(e)

    ani = FuncAnimation(fig, animate, interval=90, save_count=frames)
    if save:
        ani.save(f'centroid_with_mask_{DATASET}.mp4')
    else:
        plt.show()


# %% MAIN
if __name__ == "__main__":
    plot_centroids(save=True)
    plot_centroids_with_masks(save=True)
# def plot_perimeters(cont=True):
    # geom = GEOMETRY['perimeters']

    # if cont:
    #     _, (ax1, ax2) = plt.subplots(
    #         2, 1)
    #     for i in range(len(geom)):
    #         ar = geom[i]
    #         if len(ar) > 1:
    #             for j in ar:
    #                 ax1.plot(i, j*PX_TO_MM, 'rx')
    #         elif len(ar) == 1:
    #             ax1.plot(i, ar[0]*PX_TO_MM, 'k.')
    #         elif len(ar) == 0:
    #             ax1.plot(i, 0, 'b*')
    #         ax2.imshow(cv2.imread('HSV Frames\\Test\\' +
    #                               DATASET + '\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE))
    #         ax1.set_ylabel('Perimeter (mm)')
    #         plt.pause(.0001)
    #     plt.show()
    # else:
    #     single_droplet, single_droplet_idx = ([], [])
    #     multi_droplet, multi_droplet_idx = ([], [])
    #     zero_droplet, zero_droplet_idx = ([], [])

    #     for i in range(len(geom)):
    #         ar = geom[i]
    #         if len(ar) > 1:
    #             for j in ar:
    #                 multi_droplet.append(j*PX_TO_MM)
    #                 multi_droplet_idx.append(i)
    #         elif len(ar) == 1:
    #             single_droplet.append(ar[0]*PX_TO_MM)
    #             single_droplet_idx.append(i)
    #         elif len(ar) == 0:
    #             zero_droplet.append(0)
    #             zero_droplet_idx.append(i)
    #     plt.plot(single_droplet_idx, single_droplet, 'k-.')
    #     plt.plot(multi_droplet_idx, multi_droplet, 'rx')
    #     plt.plot(zero_droplet_idx, zero_droplet, 'b*')

    #     plt.ylim([0, 500*PX_TO_MM])
    #     plt.tight_layout()
    #     plt.savefig('perimeters.pdf')
    #     plt.show()


# # def plot_areas(cont=True):
#     geom = GEOMETRY['areas']
#     if cont:
#         _, (ax1, ax2) = plt.subplots(
#             2, 1)
#         for i in range(len(geom)):
#             ar = geom[i]
#             if len(ar) > 1:
#                 for j in ar:
#                     ax1.plot(i, j*PX_TO_MM**2, 'rx')
#             elif len(ar) == 1:
#                 ax1.plot(i, ar[0]*PX_TO_MM**2, 'k.')
#             elif len(ar) == 0:
#                 ax1.plot(i, 0, 'b*')
#             ax2.imshow(cv2.imread('HSV Frames\\Test\\' +
#                                   DATASET + '\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE))
#             ax1.set_ylabel(r'Area ($mm^2$)')
#             ax1.set_ylim([0, 500*PX_TO_MM])
#             plt.tight_layout()
#             plt.savefig(f'Graphs\\{i}.jpg', format='jpg')
#     else:

#         single_droplet, single_droplet_idx = ([], [])
#         multi_droplet, multi_droplet_idx = ([], [])
#         zero_droplet, zero_droplet_idx = ([], [])
#         dt = cp.time_period([P1, P2, P2, P2])
#         for i in range(len(geom)):
#             ar = geom[i]
#             if len(ar) == 0:
#                 single_droplet.append(0)
#             elif len(ar) > 1:
#                 single_droplet.append(sum(ar)*PX_TO_MM**2*0.01)
#             else:
#                 single_droplet.append(max(ar)*PX_TO_MM**2*0.01)
#             # if len(ar) > 1:
#             #     for j in ar:
#             #         multi_droplet.append(j*PX_TO_MM**2)
#             #         multi_droplet_idx.append(i)
#             # elif len(ar) == 1:
#             #     single_droplet.append(ar[0]*PX_TO_MM**2)
#             #     single_droplet_idx.append(i)
#             # elif len(ar) == 0:
#             #     zero_droplet.append(0)
#             #     zero_droplet_idx.append(i)
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
#         # smooth_droplet = smooth(np.array(single_droplet), 100)

#         # peaks = scipy.signal.find_peaks(-smooth_droplet, height=-.04)
#         # for i, peak in enumerate(peaks[0]):
#         #     if i == 0:
#         #         ax1.axvline(x=peak, color='b', alpha=0.4, label='detachment')
#         #     else:
#         #         ax1.axvline(x=peak, color='b', alpha=0.4)
#         ax1.plot(range(len(single_droplet)), single_droplet,
#                  'k-', alpha=.5, label='area')
#         # ax1.plot(range(len(multi_droplet_idx)), multi_droplet,
#         #          'rx', alpha=.5, label='area')
#         # ax1.plot(range(len(zero_droplet_idx)), zero_droplet,
#         #          'bx', alpha=.5, label='area')
#         # # ax1.plot(range(len(smooth_droplet)), smooth_droplet,
#         #          'k-', label='smoothed area')

#         # plt.plot(multi_droplet_idx, multi_droplet, 'rx')
#         # plt.plot(zero_droplet_idx, zero_droplet, 'b*')

#         ax1.set_ylim([0, 2200*PX_TO_MM**2*0.01])
#         ax1.set_ylabel(r'Area ($cm^2$)')
#         ax1.legend()
#         # plt.savefig('areas.pdf')

#         return ax2  # , peaks


# # def plot_fft():
#     xf, yf, N = compute_fft(plot_areas(cont=False))
#     plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]), 'k-')
#     plt.show()


# # def main():
#     plot_areas(cont=False)


# # def smooth(x, window_len=11, window='hanning'):
#     """smooth the data using a window with requested size.

#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.

#     input:
#         x: the input signal
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.

#     output:
#         the smoothed signal

#     example:

#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)

#     see also:

#     np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
#     scipy.signal.lfilter

#     TODO: the window parameter could be the window itself if an array instead of a string
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """

#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")

#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")

#     if window_len < 3:
#         return x

#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError(
#             "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

#     s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
#     # print(len(s))
#     if window == 'flat':  # moving average
#         w = np.ones(window_len, 'd')
#     else:
#         w = eval('np.'+window+'(window_len)')

#     y = np.convolve(w/w.sum(), s, mode='valid')
#     return y
