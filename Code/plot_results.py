'''
DOC
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from utils.preprocessing import rgb2gray
from utils.postprocessing import parse_centroid_coords, smooth_signal
mpl.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
sns.set()

# %% VAIRABLES

# Pixel to distance conversions
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m


ARCHITECTURE_NAME = 'unet'
DATASET = 'Spray'
N_FILTERS = 8
BATCH_SIZE_TRAIN = 8
EPOCHS = 200

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


def animate_centroids(save=False):
    '''
    Saves animation of original image with markers for the centroids of each droplet.
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


def animate_areas(save=False):
    '''
    DOC
    '''
    def areas_img_generator():
        for i, img in enumerate(IMAGES):
            yield AREAS[i], img
    data = areas_img_generator()
    areas, img = next(data)
    fig = plt.figure()
    grid_spec = fig.add_gridspec(2, 1)
    ax0 = fig.add_subplot(grid_spec[0, 0])
    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(grid_spec[1, 0])

    if len(areas) == 0:
        ax0.plot(0, 0, 'bo')
    else:
        for area in areas:
            if len(areas) > 1:
                ax0.plot(0, area, 'rx')
            elif len(areas) == 1:
                ax0.plot(0, area, 'k.')

    image_ax = ax1.imshow(img)

    ax0.set_ylim([0, max(max(AREAS))])

    original_area = []
    for areas in AREAS:
        if len(areas) > 0:
            original_area.append(sum(areas))
        else:
            original_area.append(0)
    smooth_area = smooth_signal(np.array(original_area), 51)
    peaks = find_peaks(-smooth_area, width=50,
                       prominence=300, height=-2500)

    def animate(i):
        areas, img = next(data)
        if len(areas) == 0:
            ax0.plot(i, 0, 'bo')
        else:
            for area in areas:
                if len(areas) > 1:
                    ax0.plot(i, area, 'rx')
                elif len(areas) == 1:
                    ax0.plot(i, area, 'k.')
        if i in peaks[0]:
            ax0.axvline(x=i, color='k', alpha=0.4, linestyle='--')
        with sns.axes_style('dark'):
            image_ax.set_data(img)
        plt.title(f'Frame #{i}')
        return image_ax,

    ani = FuncAnimation(fig, animate, save_count=N_FRAMES-1)
    if save:
        ani.save(os.path.join('Output', 'Videos',
                              f'areas_{DATASET.lower()}.mp4'), writer=WRITER)
    else:
        plt.show()


def animate_areas_with_masks(save=False):
    '''
    DOC
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
    with sns.axes_style('dark'):
        ax1 = fig.add_subplot(grid_spec[1, 0])
        ax2 = fig.add_subplot(grid_spec[1, 1])

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

    def animate(i):
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
        plt.title(f'Frame #{i}')
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
            plt.title(f'Frame #{i}')
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
    '''
    cent = GEOMETRY['centroids'][num]
    area = GEOMETRY['areas'][num]

    image = DATA_RGB['images'][num]
    mask = DATA_MASKS['preds'][num]

    max_area = np.argmax(area)
    height = cent[max_area][1]

    strip = rgb2gray(image[height, :])
    mask_strip = mask[height, :]

    fig1 = plt.figure()
    with sns.axes_style('dark'):
        ax1 = plt.axes()
    ax1.imshow(image)
    ax1.plot(strip, 'k', linewidth=1)
    ax1.plot(mask_strip, 'r', linewidth=1)
    ax1.plot([0, strip.shape[0]-2], [height, height], linewidth=1)
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
    plt.title(f'Frame #{num}')
    plt.tight_layout()

    if save:
        fig1.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{num}_1.pdf'), format='pdf')
        fig2.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{num}_2.pdf'), format='pdf')
        fig3.savefig(os.path.join('Output', 'Plots',
                                  f'boundary_{DATASET.lower()}_{num}_3.pdf'), format='pdf')
    else:
        plt.show()


def split_areas():
    '''
    DOC
    '''
    single_droplet = []
    single_droplet_idx = []
    multi_droplet = []
    multi_droplet_idx = []
    zero_droplet = []
    zero_droplet_idx = []

    for i, areas in enumerate(AREAS):
        if len(areas) == 0:
            zero_droplet.append(0)
            zero_droplet_idx.append(i)
        elif len(areas) == 1:
            single_droplet.append(areas[0])
            single_droplet_idx.append(i)
        else:
            multi_droplet.append(tuple(areas))
            multi_droplet_idx.append(i)

    return single_droplet, single_droplet_idx, zero_droplet, zero_droplet_idx, multi_droplet, multi_droplet_idx


def plot_smooth_area():
    '''
    DOC
    '''
    original_area = []
    for areas in AREAS:
        if len(areas) > 0:
            original_area.append(np.mean(areas))
        else:
            original_area.append(0)
    _, axes = plt.subplots(1, 1)
    single, single_id, zero, zero_id, multi, multi_id = split_areas()
    axes.scatter(single_id, single, alpha=0.3, c='black', s=5)
    axes.scatter(zero_id, zero, alpha=0.3, c='blue', s=5)
    for i, j in zip(multi_id, multi):
        plt.scatter([i]*len(j), j, alpha=0.3, c='red', s=5, marker='x')
    smooth_area = smooth_signal(np.array(original_area), 51, window='hamming')
    axes.plot(smooth_area, label='Smooth')
    peaks = find_peaks(-smooth_area, width=50,
                       prominence=300, height=-2500)
    for i, peak in enumerate(peaks[0]):
        if i == 0:
            axes.axvline(x=peak, color='k', alpha=0.4,
                         linestyle='--', label='Detachment')
        else:
            axes.axvline(x=peak, color='k', alpha=0.4, linestyle='--')

    plt.show()


# %% MAIN
if __name__ == "__main__":
    # animate_centroids(save=True)
    animate_centroids_with_masks(save=True)
    # animate_areas(save=True)
    # animate_areas_with_masks(save=True)
    # plot_smooth_area()

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
