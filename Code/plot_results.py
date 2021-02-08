'''
DOC
'''
# %% IMPORTS
import os
import sys
import pickle
import numpy as np
import scipy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from cv2 import cv2
from progressbar import progressbar as progress
from utils.preprocessing import rgb2gray
mpl.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
sns.set()

# %% VAIRABLES
'''
Pixel to distance conversions
26 px = .045 in = 1.143 mm
1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
'''

ARCHITECTURE_NAME = 'unet'
DATASET = 'Globular'
N_FILTERS = 16
BATCH_SIZE_TRAIN = 16
EPOCHS = 100

DATA_DIR_RGB = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz')
DATA_DIR_GRAY = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_gray.npz')
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')
GEOMETRY_DIR = os.path.join('Output', 'Geometry',
                            f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle')

P1 = 334*10 ** (-6)  # period between frames in seconds
P2 = 333*10 ** (-6)
PX_TO_MM = 4.396153846153846 * 10**(-2)

DATA_RGB = np.load(DATA_DIR_RGB)
DATA_GRAY = np.load(DATA_DIR_GRAY)
DATA_MASKS = np.load(PREDS_DIR)

IMAGES = DATA_RGB['images']
MASKS = DATA_MASKS['preds']
N_FRAMES = len(IMAGES)

with open(GEOMETRY_DIR, 'rb') as f:
    GEOMETRY = pickle.load(f)

CENTROIDS = GEOMETRY['centroids']
AREAS = GEOMETRY['areas']

WRITER = animation.writers['ffmpeg'](fps=30, bitrate=1800)
# %% FUNCTIONS


def parse_centroid_coords(cent):
    x = []
    y = []
    for c in cent:
        x.append(c[0])
        y.append(c[1])
    return x, y


def centroid_img_generator():
    for cent, image in zip(CENTROIDS, IMAGES):
        yield cent, image


def plot_centroids(save=False):
    data = centroid_img_generator()
    cent, img = next(data)
    x, y = parse_centroid_coords(cent)
    fig = plt.figure()
    with sns.axes_style('dark'):
        ax = plt.axes()
    im = ax.imshow(img)
    if len(cent) > 1:
        single_droplet, = ax.plot([], [], 'k.')
        multi_droplet, = ax.plot(x, y, 'rx')
    elif len(cent) == 1:
        single_droplet, = ax.plot(x, y, 'k.')
        multi_droplet, = ax.plot([], [], 'rx')
    else:
        single_droplet, = ax.plot([], [], 'k.')
        multi_droplet, = ax.plot([], [], 'rx')

    def animate(i):
        try:
            cent, img = next(data)
            im.set_data(img)
            x, y = parse_centroid_coords(cent)
            if len(cent) > 1:
                single_droplet.set_data([], [])
                multi_droplet.set_data(x, y)
            elif len(cent) == 1:
                single_droplet.set_data(x, y)
                multi_droplet.set_data([], [])
            else:
                single_droplet.set_data([], [])
                multi_droplet.set_data([], [])

            return im, single_droplet, multi_droplet,
        except StopIteration as e:
            print(e)

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'centroids_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def centroid_mask_img_generator():
    for cent, image, mask in zip(CENTROIDS, IMAGES, MASKS):
        yield cent, image, mask


def plot_centroids_with_masks(save=False):
    data = centroid_mask_img_generator()
    cent, img, msk = next(data)
    x, y = parse_centroid_coords(cent)
    with sns.axes_style('dark'):
        fig, ax = plt.subplots(1, 2)
    im = ax[0].imshow(img)
    mk = ax[1].imshow(msk)
    if len(cent) > 1:
        single_droplet_1, = ax[0].plot([], [], 'k.')
        multi_droplet_1, = ax[0].plot(x, y, 'rx')

        single_droplet_2, = ax[1].plot([], [], 'k.')
        multi_droplet_2, = ax[1].plot(x, y, 'rx')
    elif len(cent) == 1:
        single_droplet_1, = ax[0].plot(x, y, 'k.')
        multi_droplet_1, = ax[0].plot([], [], 'rx')

        single_droplet_2, = ax[1].plot(x, y, 'k.')
        multi_droplet_2, = ax[1].plot([], [], 'rx')

    else:
        single_droplet_1, = ax[0].plot([], [], 'k.')
        multi_droplet_1, = ax[0].plot([], [], 'rx')

        single_droplet_2, = ax[1].plot([], [], 'k.')
        multi_droplet_2, = ax[1].plot([], [], 'rx')

    def animate(i):
        try:
            cent, img, msk = next(data)
            im.set_data(img)
            mk.set_data(msk)
            x, y = parse_centroid_coords(cent)
            if len(cent) > 1:
                single_droplet_1.set_data([], [])
                single_droplet_2.set_data([], [])
                multi_droplet_1.set_data(x, y)
                multi_droplet_2.set_data(x, y)
            elif len(cent) == 1:
                single_droplet_1.set_data(x, y)
                single_droplet_2.set_data(x, y)
                multi_droplet_1.set_data([], [])
                multi_droplet_2.set_data([], [])
            else:
                single_droplet_1.set_data([], [])
                single_droplet_2.set_data([], [])
                multi_droplet_1.set_data([], [])
                multi_droplet_2.set_data([], [])

            return im, mk, single_droplet_1, multi_droplet_1, single_droplet_2, multi_droplet_2,

        except StopIteration as e:
            print(e)

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'centroid_with_masks_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def plot_strip(n, save=False):
    cent = GEOMETRY['centroids'][n]
    area = GEOMETRY['areas'][n]

    image = DATA_RGB['images'][n]
    mask = DATA_MASKS['preds'][n]

    m = np.argmax(area)
    h = cent[m][1]

    strip = rgb2gray(image[h, :])
    mask_strip = mask[h, :]

    fig1 = plt.figure()
    with sns.axes_style('dark'):
        ax1 = plt.axes()
    ax1.imshow(image)
    ax1.plot(strip, 'k', linewidth=1)
    ax1.plot(mask_strip, 'r', linewidth=1)
    ax1.plot([0, strip.shape[0]-2], [h, h], linewidth=1)
    plt.tight_layout()

    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(strip, linewidth=1, label='real pixel value')
    ax2.plot(mask_strip, linewidth=1, label='prediction')
    ax2.set_xlim([0, strip.shape[0]-2])
    ax2.legend()
    plt.tight_layout()

    fig3 = plt.figure()
    with sns.axes_style('dark'):
        ax3 = plt.axes()
    ax3.imshow(mask)
    plt.tight_layout()

    if save:
        fig1.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{n}_1.pdf'), format='pdf')
        fig2.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{n}_2.pdf'), format='pdf')
        fig3.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{n}_3.pdf'), format='pdf')
    else:
        plt.show()


def areas_img_generator():
    for i, img in enumerate(IMAGES):
        yield AREAS[i], img


def plot_areas(save=False):
    data = areas_img_generator()
    area, img = next(data)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(gs[1, 0])

    if len(area) == 0:
        ax0.plot(0, 0, 'bo')
    else:
        for a in area:
            if len(area) > 1:
                ax0.plot(0, a, 'rx')
            elif len(area) == 1:
                ax0.plot(0, a, 'k.')

    im = ax1.imshow(img)

    ax0.set_ylim([0, max(max(AREAS))])

    def animate(i):
        area, img = next(data)
        if len(area) == 0:
            ax0.plot(i, 0, 'bo')
        else:
            for a in area:
                if len(area) > 1:
                    ax0.plot(i, a, 'rx')
                elif len(area) == 1:
                    ax0.plot(i, a, 'k.')
        with sns.axes_style('dark'):
            im.set_data(img)
        return im,

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'areas_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def areas_centroid_img_mask_generator():
    for i, cent_img_mask in enumerate(zip(CENTROIDS, IMAGES, MASKS)):
        yield AREAS[i], *cent_img_mask


def plot_areas_with_masks(save=False):
    data = areas_centroid_img_mask_generator()
    area, cent, img, mask = next(data)
    x, y = parse_centroid_coords(cent)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, :])
    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

    if len(area) == 0:
        ax0.plot(0, 0, 'bo')
    else:
        for a in area:
            if len(area) > 1:
                ax0.plot(0, a, 'rx')
            elif len(area) == 1:
                ax0.plot(0, a, 'k.')

    im = ax1.imshow(img)
    mk = ax2.imshow(mask)
    ax0.set_ylim([0, max(max(AREAS))])

    if len(cent) > 1:
        single_droplet_1, = ax1.plot([], [], 'k.')
        multi_droplet_1, = ax1.plot(x, y, 'rx')

        single_droplet_2, = ax2.plot([], [], 'k.')
        multi_droplet_2, = ax2.plot(x, y, 'rx')
    elif len(cent) == 1:
        single_droplet_1, = ax1.plot(x, y, 'k.')
        multi_droplet_1, = ax1.plot([], [], 'rx')

        single_droplet_2, = ax2.plot(x, y, 'k.')
        multi_droplet_2, = ax2.plot([], [], 'rx')

    else:
        single_droplet_1, = ax1.plot([], [], 'k.')
        multi_droplet_1, = ax1.plot([], [], 'rx')

        single_droplet_2, = ax2.plot([], [], 'k.')
        multi_droplet_2, = ax2.plot([], [], 'rx')

    def animate(i):
        area, cent, img, mask = next(data)
        if len(area) == 0:
            ax0.plot(i, 0, 'bo')
        else:
            for a in area:
                if len(area) > 1:
                    ax0.plot(i, a, 'rx')
                elif len(area) == 1:
                    ax0.plot(i, a, 'k.')
        im.set_data(img)
        mk.set_data(mask)

        x, y = parse_centroid_coords(cent)
        if len(cent) > 1:
            single_droplet_1.set_data([], [])
            single_droplet_2.set_data([], [])
            multi_droplet_1.set_data(x, y)
            multi_droplet_2.set_data(x, y)
        elif len(cent) == 1:
            single_droplet_1.set_data(x, y)
            single_droplet_2.set_data(x, y)
            multi_droplet_1.set_data([], [])
            multi_droplet_2.set_data([], [])
        else:
            single_droplet_1.set_data([], [])
            single_droplet_2.set_data([], [])
            multi_droplet_1.set_data([], [])
            multi_droplet_2.set_data([], [])

        return im, mk, single_droplet_1, multi_droplet_1, single_droplet_2, multi_droplet_2,

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'areas_with_masks_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


# %% MAIN
if __name__ == "__main__":
    plot_centroids(save=True)
    plot_centroids_with_masks(save=True)
    plot_areas(save=True)
    plot_areas_with_masks(save=True)
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
