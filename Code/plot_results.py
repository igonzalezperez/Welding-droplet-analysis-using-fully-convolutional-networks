'''
Functions that output plots and animations of
relevant geometric and kinematic droplet properties
using predicted segmentation maps.
'''
# %% IMPORTS
import os
import pickle
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Ellipse
from scipy.signal import find_peaks, hann
from utils.postprocessing import parse_centroid_coords, smooth_signal, set_size, latex_plot_config, axvlines
from utils.misc import get_concat_h, chunks

# video settings
matplotlib.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
# set style
sns.set()

# %% VAIRABLES
# Pixel to distance conversions
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
#
# LaTex \textwidth = 472.03123 pt. Useful for figure sizing

ARCHITECTURE_NAME = 'unet'
DATASET = 'Globular'
N_FILTERS = 32
BATCH_SIZE_TRAIN = 16
EPOCHS = 200

DATA_DIR_RGB = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz')
DATA_DIR_GRAY = os.path.join(
    'Data', 'Image', 'Input', f'{DATASET.lower()}_gray.npz')
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')
GEOMETRY_DIR = os.path.join('Output', 'Geometry',
                            f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle')

FPS = 3000.0
PERIOD = 1/FPS
PX_TO_MM = 4.396153846153846 * 10**(-2)

DATA_RGB = np.load(DATA_DIR_RGB)
DATA_GRAY = np.load(DATA_DIR_GRAY)
DATA_MASKS = np.load(PREDS_DIR)
OFFSET = 0
IMAGES = DATA_RGB['images'][OFFSET:]
MASKS = DATA_MASKS['preds'][OFFSET:]

N_FRAMES = len(IMAGES)

with open(GEOMETRY_DIR, 'rb') as f:
    GEOMETRY = pickle.load(f)

CENTROIDS = GEOMETRY['centroids'][OFFSET:]
AREAS = GEOMETRY['areas'][OFFSET:]

WRITER = animation.writers['ffmpeg'](fps=30, bitrate=1800)
# %% FUNCTIONS


def plot_centroids(num, save=False):
    '''
    Plots a specific frame of a video and overlays the predicted centroid.

    Args:
    num {int} -- mumber of the frame to plot.

    Kwargs:
    save {bool} -- wether to save the plot as .pgf file or to show using plt.show().
    '''
    cent = CENTROIDS[num]
    x_coord, y_coord = parse_centroid_coords(cent)
    image = Image.fromarray(IMAGES[num])
    mask = Image.fromarray(MASKS[num])
    img_mask = get_concat_h(image, mask)
    with sns.axes_style('dark'):
        fig, axes = plt.subplots(1, 1, figsize=set_size(
            fraction=1, aspect_ratio=IMAGES.shape[1]/(2*IMAGES.shape[2]), subplots=(1, 1)))
    axes.set_xticks([])
    axes.set_yticks([])

    axes.set_title(f'Frame {num}')
    axes.imshow(img_mask)

    if len(cent) > 1:
        axes.scatter(x_coord, y_coord, s=10, marker='x', color='r')
        axes.scatter([x+IMAGES.shape[2] for x in x_coord],
                     y_coord, s=10, marker='x', color='r')

    elif len(cent) == 1:
        axes.scatter(x_coord, y_coord, s=10, marker='.', color='k')
        axes.scatter([x+IMAGES.shape[2] for x in x_coord],
                     y_coord, s=10, marker='.', color='k')

    if save:
        os.makedirs(os.path.join('Output', 'Plots',
                                 'centroid_samples'), exist_ok=True)
        fig.tight_layout()
        latex_plot_config()
        fig.savefig(os.path.join('Output', 'Plots',
                                 'centroid_samples', f'{DATASET.lower()}_{num}.pgf'))
    else:
        plt.show()


def animate_centroids(save=False):
    '''
    Saves animation of original image with markers for the centroids of each droplet.

    Kwargs:
    save {bool} -- wether to save the animation as a .mp4 or show it using plt.show().
    '''
    def centroid_img_generator():
        for cent, image in zip(CENTROIDS, IMAGES):
            yield cent, image
    data = centroid_img_generator()
    cent, img = next(data)
    x_coord, y_coord = parse_centroid_coords(cent)
    fig = plt.figure()
    with sns.axes_style('dark'):
        axes = plt.axes()
    image_ax = axes.imshow(img)
    if len(cent) > 1:
        single_droplet, = axes.plot([], [], 'k.')
        multi_droplet, = axes.plot(x_coord, y_coord, 'rx')
    elif len(cent) == 1:
        single_droplet, = axes.plot(x_coord, y_coord, 'k.')
        multi_droplet, = axes.plot([], [], 'rx')
    else:
        single_droplet, = axes.plot([], [], 'k.')
        multi_droplet, = axes.plot([], [], 'rx')
    plt.show()

    def animate(i):
        try:
            cent, img = next(data)
            image_ax.set_data(img)
            x_coord, y_coord = parse_centroid_coords(cent)
            if len(cent) > 1:
                single_droplet.set_data([], [])
                multi_droplet.set_data(x_coord, y_coord)
            elif len(cent) == 1:
                single_droplet.set_data(x_coord, y_coord)
                multi_droplet.set_data([], [])
            else:
                single_droplet.set_data([], [])
                multi_droplet.set_data([], [])
            plt.title(f'Frame #{i}')
            return image_ax, single_droplet, multi_droplet,
        except StopIteration as error:
            print(error)

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'centroids_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def plot_areas(width, func,  save=False):
    '''
    Plots area of a droplet over time for consecutive time frames. Black dots represent one droplet, red crosses are for multiple
    droplets in the same frame and blue dots means no droplet is detected (area=0). Also, the detachment of droplets is shown in
    dashed vertical lines obtained by finding the (minimum) peaks of a smoothed version (green curve) of the original signal.
    Smoothing is done using a hanning window over the data. Datapoints that have multiple values are summed.

    Args:
    width {int} -- length of the time frames, i.e. a signal with length 100 plotted with width 10 will yield 10 consecutive plots.

    Kwargs:
    save {bool} -- wether to save each plot as a .pgf file or show it using plt.show().
    '''
    if save:
        latex_plot_config()
    areas_chunks = chunks(AREAS, width)

    for k, chunk in enumerate(areas_chunks):
        fig, axes = plt.subplots(1, 1, figsize=set_size(aspect_ratio=.5))
        axes.set_xlabel('Frame')
        axes.set_ylabel(r'Area [$mm^2$]')
        axes.set_ylim([-.1, max(max(chunk))+1])
        single_id, single = ([], [])
        multi_id, multi = ([], [])
        zero_id = []
        original_area_id, original_area = ([], [])
        for i, areas in enumerate(chunk):
            i = i+OFFSET
            if len(areas) > 0:
                original_area.append(func(areas))
            else:
                original_area.append(0)
            original_area_id.append(i+width*k)

            if len(areas) == 0:
                zero_id.append(i + width*k)
            else:
                for area in areas:
                    if len(areas) == 1:
                        single_id.append(i+width*k)
                        single.append(area)
                    else:
                        multi_id.append(i+width*k)
                        multi.append(area)
        # globular
        smooth_area = smooth_signal(np.array(original_area), 41)
        peaks = find_peaks(-smooth_area, height=-5, prominence=2)[0]
        # spray
        # smooth_area = smooth_signal(np.array(original_area), 7)
        # peaks = find_peaks(smooth_area)[0]
        # print(len(peaks))
        # circ = Ellipse(xy=(9600, 5), height=10, width=300, fill=False,
        #                color='k', linewidth=1)
        # axes.add_artist(circ)
        axes.scatter(zero_id, np.zeros(len(zero_id)), s=12,
                     marker='o', color='b', label='Zero droplets')
        axes.scatter(single_id, single, s=12, marker='.',
                     color='k', label='Single droplet')
        axes.scatter(multi_id, multi, s=9, marker='x',
                     color='r', label='Multiple droplets')
        axvlines(xs=peaks+width*k+OFFSET, ax=axes, label='Detachment',
                 linestyle='--', color='k', alpha=0.6)
        axes.plot(original_area_id,
                  smooth_area, 'g-', label='Smoothed')
        # axes.legend()
        if save:
            fig.tight_layout()
            fig.savefig(os.path.join('Output', 'Plots', 'areas',
                                     f'{DATASET.lower()}_area_{k}.pgf'))
            plt.close(fig=fig)
        else:
            plt.show()


def animate_areas(width, func, save=False):
    '''
    Animates area of a droplet over time for consecutive time frames. Black dots represent one droplet, red crosses are for multiple
    droplets in the same frame and blue dots means no droplet is detected (area=0). Also, the detachment of droplets is shown in
    dashed vertical lines obtained by finding the (minimum) peaks of a smoothed version of the original signal.
    Smoothing is done using a hanning window over the data. Datapoints that have multiple values are summed.
    Below the graph, the corresponding image for that frame is displayed.

    Args:
    width {int} -- length of the time frames, e.g. a signal with length 100 animated with width 10 will refresh the animation
                   every 10 frames.
    func {python function} -- Which function to use to reduce multiple values of area (e.g. max, sum, mean)

    Kwargs:
    save {bool} -- wether to save the animation as a .mp4 file or show it using plt.show().
    '''
    def areas_img_generator():
        for i, img in enumerate(IMAGES):
            yield AREAS[i], img
    data = areas_img_generator()
    fig = plt.figure()
    grid_spec = fig.add_gridspec(2, 1)

    ax0 = fig.add_subplot(grid_spec[0, 0])
    ax0.xaxis.tick_top()
    ax0.set_xlabel('Frame')
    ax0.xaxis.set_label_position('top')
    ax0.set_ylabel(r'Area [$mm^2$]')

    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(grid_spec[1, 0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    image_ax = ax1.imshow(np.zeros(IMAGES[0].shape))

    original_area = []
    for areas in AREAS:
        if len(areas) > 0:
            original_area.append(func(areas))
        else:
            original_area.append(0)
    # globular
    smooth_area = smooth_signal(np.array(original_area), 41)
    peaks = find_peaks(-smooth_area, height=-5, prominence=2)[0]
    # spray
    # smooth_area = smooth_signal(np.array(original_area), 7)
    # peaks = find_peaks(smooth_area)[0]

    def init():
        image_ax.set_data(np.zeros(IMAGES[0].shape))
        return image_ax,

    def animate(i):
        print(i)
        if i != 0 and i % width == 0:
            ax0.clear()
            ax0.xaxis.tick_top()
            ax0.set_xlabel('Frame')
            ax0.xaxis.set_label_position('top')
            ax0.set_ylabel(r'Area [$mm^2$]')
        areas, img = next(data)
        if len(areas) == 0:
            ax0.plot(i, 0, 'bo')
        else:
            for area in areas:
                if len(areas) > 1:
                    ax0.plot(i, area, 'rx')
                elif len(areas) == 1:
                    ax0.plot(i, area, 'k.')
        if i in peaks:
            ax0.axvline(x=i, color='k', alpha=0.4, linestyle='--')
        with sns.axes_style('dark'):
            image_ax.set_data(img)
        plt.title(f'Frame {i}')
        return image_ax,

    ani = FuncAnimation(fig, animate, init_func=init, save_count=N_FRAMES)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'areas_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def animate_areas_with_masks(width, func, save=False):
    '''
    Same as animate_areas() but includes the corresponding masks. Also displays the centroids.

    Args:
    width {int} -- length of the time frames, e.g. a signal with length 100 animated with width 10 will refresh the animation
                   every 10 frames.

    Kwargs:
    save {bool} -- wether to save the animation as a .mp4 file or show it using plt.show().
    '''
    def areas_centroid_img_mask_generator():
        for i, cent_img_mask in enumerate(zip(CENTROIDS, IMAGES, MASKS)):
            yield AREAS[i], *cent_img_mask
    data = areas_centroid_img_mask_generator()
    areas, cent, img, mask = next(data)
    x_coord, y_coord = parse_centroid_coords(cent)
    fig = plt.figure()
    grid_spec = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(grid_spec[0, :])
    ax0.xaxis.tick_top()
    ax0.set_xlabel('Frame')
    ax0.xaxis.set_label_position('top')
    ax0.set_ylabel(r'Area [$mm^2$]')

    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(grid_spec[1, 0])
        ax2 = fig.add_subplot(grid_spec[1, 1])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    if len(areas) == 0:
        ax0.plot(0, 0, 'bo')
    else:
        for area in areas:
            if len(areas) > 1:
                ax0.plot(0, area, 'rx')
            elif len(areas) == 1:
                ax0.plot(0, area, 'k.')

    image_ax = ax1.imshow(img)
    mask_ax = ax2.imshow(mask)
    ax0.set_ylim([0, max(max(AREAS))])

    if len(cent) > 1:
        single_droplet_1, = ax1.plot([], [], 'k.')
        multi_droplet_1, = ax1.plot(x_coord, y_coord, 'rx')

        single_droplet_2, = ax2.plot([], [], 'k.')
        multi_droplet_2, = ax2.plot(x_coord, y_coord, 'rx')
    elif len(cent) == 1:
        single_droplet_1, = ax1.plot(x_coord, y_coord, 'k.')
        multi_droplet_1, = ax1.plot([], [], 'rx')

        single_droplet_2, = ax2.plot(x_coord, y_coord, 'k.')
        multi_droplet_2, = ax2.plot([], [], 'rx')

    else:
        single_droplet_1, = ax1.plot([], [], 'k.')
        multi_droplet_1, = ax1.plot([], [], 'rx')

        single_droplet_2, = ax2.plot([], [], 'k.')
        multi_droplet_2, = ax2.plot([], [], 'rx')
    original_area = []
    for areas in AREAS:
        if len(areas) > 0:
            original_area.append(func(areas))
        else:
            original_area.append(0)
    # globular
    # smooth_area = smooth_signal(np.array(original_area), 41)
    # peaks = find_peaks(-smooth_area, height=-5, prominence=2)[0]
    # spray
    smooth_area = smooth_signal(np.array(original_area), 7)
    peaks = find_peaks(smooth_area)[0]

    def animate(i):
        print(i)
        if i != 0 and i % width == 0:
            ax0.clear()
            ax0.xaxis.tick_top()
            ax0.set_xlabel('Frame')
            ax0.xaxis.set_label_position('top')
            ax0.set_ylabel(r'Area [$mm^2$]')
        areas, cent, img, mask = next(data)
        if len(areas) == 0:
            ax0.plot(i, 0, 'bo')
        else:
            for area in areas:
                if len(areas) > 1:
                    ax0.plot(i, area, 'rx')
                elif len(areas) == 1:
                    ax0.plot(i, area, 'k.')
        image_ax.set_data(img)
        mask_ax.set_data(mask)

        x_coord, y_coord = parse_centroid_coords(cent)
        if len(cent) > 1:
            single_droplet_1.set_data([], [])
            single_droplet_2.set_data([], [])
            multi_droplet_1.set_data(x_coord, y_coord)
            multi_droplet_2.set_data(x_coord, y_coord)
        elif len(cent) == 1:
            single_droplet_1.set_data(x_coord, y_coord)
            single_droplet_2.set_data(x_coord, y_coord)
            multi_droplet_1.set_data([], [])
            multi_droplet_2.set_data([], [])
        else:
            single_droplet_1.set_data([], [])
            single_droplet_2.set_data([], [])
            multi_droplet_1.set_data([], [])
            multi_droplet_2.set_data([], [])
        if i in peaks:
            ax0.axvline(x=i, color='k', alpha=0.4, linestyle='--')
        ax1.set_title(f'Frame {i}')
        return image_ax, mask_ax, single_droplet_1, multi_droplet_1, single_droplet_2, multi_droplet_2,

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'areas_with_masks_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def animate_centroids_with_masks(save=False):
    '''
    Saves animation of original image and mask with markers for the centroids of each droplet.

    Kwargs:
    save {bool} -- wether to save the animation as a .mp4 or to show it with plt.show().
    '''
    def centroid_mask_img_generator():
        for cent, image, mask in zip(CENTROIDS, IMAGES, MASKS):
            yield cent, image, mask
    data = centroid_mask_img_generator()
    cent, img, msk = next(data)
    x_coord, y_coord = parse_centroid_coords(cent)
    with sns.axes_style('dark'):
        fig, axes = plt.subplots(1, 2)
    image_ax = axes[0].imshow(img)
    mask_ax = axes[1].imshow(msk)
    if len(cent) > 1:
        single_droplet_1, = axes[0].plot([], [], 'k.')
        multi_droplet_1, = axes[0].plot(x_coord, y_coord, 'rx')

        single_droplet_2, = axes[1].plot([], [], 'k.')
        multi_droplet_2, = axes[1].plot(x_coord, y_coord, 'rx')
    elif len(cent) == 1:
        single_droplet_1, = axes[0].plot(x_coord, y_coord, 'k.')
        multi_droplet_1, = axes[0].plot([], [], 'rx')

        single_droplet_2, = axes[1].plot(x_coord, y_coord, 'k.')
        multi_droplet_2, = axes[1].plot([], [], 'rx')

    else:
        single_droplet_1, = axes[0].plot([], [], 'k.')
        multi_droplet_1, = axes[0].plot([], [], 'rx')

        single_droplet_2, = axes[1].plot([], [], 'k.')
        multi_droplet_2, = axes[1].plot([], [], 'rx')

    def animate(i):
        try:
            cent, img, msk = next(data)
            image_ax.set_data(img)
            mask_ax.set_data(msk)
            x_coord, y_coord = parse_centroid_coords(cent)
            if len(cent) > 1:
                single_droplet_1.set_data([], [])
                single_droplet_2.set_data([], [])
                multi_droplet_1.set_data(x_coord, y_coord)
                multi_droplet_2.set_data(x_coord, y_coord)
            elif len(cent) == 1:
                single_droplet_1.set_data(x_coord, y_coord)
                single_droplet_2.set_data(x_coord, y_coord)
                multi_droplet_1.set_data([], [])
                multi_droplet_2.set_data([], [])
            else:
                single_droplet_1.set_data([], [])
                single_droplet_2.set_data([], [])
                multi_droplet_1.set_data([], [])
                multi_droplet_2.set_data([], [])
            plt.title(f'Frame {i}')
            return image_ax, mask_ax, single_droplet_1, multi_droplet_1, single_droplet_2, multi_droplet_2,

        except StopIteration as error:
            print(error)

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'centroid_with_masks_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def plot_strip(num, save=False):
    '''
    Plots horizontal strip of an image's pixel values and compares it to mask prediction.

    Args:
    num {int} -- number of frame to plot.

    Kwargs:
    save {bool} -- wether to save the plot as a .pgf or display it with plt.show().
    '''
    cent = GEOMETRY['centroids'][num]
    area = GEOMETRY['areas'][num]

    image = DATA_GRAY['images'][num]
    mask = DATA_MASKS['preds'][num]

    max_area = np.argmax(area)
    height = cent[max_area][1]

    strip = image[height, :].astype('float64')
    mask_strip = mask[height, :].astype('float64')

    fig = plt.figure(figsize=set_size(
        472.03123, 1, aspect_ratio=1, subplots=(4, 4)))
    gsc = gridspec.GridSpec(2, 2, figure=fig)

    with sns.axes_style('dark'):
        ax0 = fig.add_subplot(gsc[0, 0])

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlim([0, image.shape[1]])
    ax0.set_ylim([image.shape[0], 0])
    ax0.imshow(image, cmap='viridis')
    ax0.plot(-strip+image.shape[0]-2, 'k', linewidth=1)
    ax0.plot(-mask_strip+image.shape[0]-2, 'r', linewidth=1)
    ax0.plot([0, strip.shape[0]-2], [height, height],
             linewidth=1, color='yellow', linestyle='--')
    ax0.set_title(f'Frame {num}')
    ax0.text(0.5, -0.1, "(a)", size=12, ha="center",
             transform=ax0.transAxes)

    ax1 = fig.add_subplot(gsc[1, 0])

    ax1.plot(strip, linewidth=1, label='real pixel value', color='k')
    ax1.plot(mask_strip, linewidth=1, label='prediction', color='r')
    ax1.set_xlim([0, strip.shape[0]-2])
    ax1.text(0.5, -0.2, "(c)", size=12, ha="center",
             transform=ax1.transAxes)
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('Pixel value')
    ax1.xaxis.tick_top()
    ax1.legend()
    with sns.axes_style('dark'):
        ax2 = fig.add_subplot(gsc[0, 1])

    ax2.imshow(mask, cmap='gray')
    ax2.plot([0, strip.shape[0]-2], [height, height],
             linewidth=1, color='yellow', linestyle='--')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.5, -0.1, "(b)", size=12, ha="center",
             transform=ax2.transAxes)

    fig.tight_layout()
    if save:
        os.makedirs(os.path.join('Output', 'Plots',
                                 f'boundary_{DATASET.lower()}'), exist_ok=True)
        fig.savefig(os.path.join('Output', 'Plots', f'boundary_{DATASET.lower()}',
                                 f'{num}.pgf'), bbox_inches='tight')
    else:
        plt.show()


def plot_hanning(save=False):
    '''
    Shows a hanning window and a noisy sine wave smoothed by a hanning window through 1D convolution.
    '''
    window = hann(51)
    fig, axes = plt.subplots(1, 2, figsize=set_size(aspect_ratio=.3))
    axes[0].plot(window)
    axes[0].set_ylabel('Amplitude')
    axes[1].set_ylabel('Amplitude')
    x_val = np.linspace(0, 8, 100)
    y_val = np.abs(np.sin(x_val))
    y_noise = y_val+np.random.randn(len(x_val))*0.1
    y_smooth = smooth_signal(y_noise)
    axes[1].plot(x_val, y_noise, label='Noisy signal')
    axes[1].plot(x_val, y_smooth, label='Smoothed signal')
    plt.legend()
    if save:
        latex_plot_config()
        fig.savefig(os.path.join('Output', 'Plots', 'hanning.pgf'))
    else:
        plt.show()


# %% MAIN
if __name__ == "__main__":
    animate_areas(width=1000, func=sum, save=True)
