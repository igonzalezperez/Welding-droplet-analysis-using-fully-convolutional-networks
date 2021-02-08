'''
Compute mask properties
'''
# %% IMPORTS
import os
import pickle
import cv2
import numpy as np
from numpy import argmax, mean, diff, log, nonzero
import matplotlib.pyplot as plt
from progressbar import progressbar as progress
from scipy.spatial import distance
from PIL import Image, ImageDraw
from scipy.signal import blackmanharris, correlate
from cv2 import cv2
from numpy import polyfit, arange

# %% VARIABLES
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
PX_TO_MM = 4.396153846153846 * 10**(-2)
P1 = 334*10 ^ (-6)  # period between frames in [s]
P2 = 333*10 ^ (-6)

ARCHITECTURE_NAME = 'unet'
DATASET = 'Globular'
N_FILTERS = 16
BATCH_SIZE_TRAIN = 16
EPOCHS = 100
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')

# %% FUNCTIONS


def compute_properties(img):
    '''
    Returns centroid, perimeter and area for every contour in an image as a list.
    img (ndarray): Image segmentation map.
    '''
    _, thresh = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    area = []
    perimeter = []
    for cnt in contours:
        mmt = cv2.moments(cnt)
        if mmt['m00'] > 20:  # ignore really small contours
            try:
                cx = int(mmt['m10']/mmt['m00'])
                cy = int(mmt['m01']/mmt['m00'])

                centroids.append((cx, cy))

                area.append(mmt['m00'])
                perimeter.append(cv2.arcLength(cnt, True))
            except ZeroDivisionError:
                continue
    return centroids, area, perimeter


def get_concat_h(im1, im2):
    '''
    Horizontaly join two images of same height.
    im1 (PIL Image object)
    im2 (PIL Image object)
    '''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# def draw_centroid(cents, image):
#     '''
#     doc
#     '''
#     line_len = 3
#     image_c = image
#     for c in cents:
#         line_1 = [(c[0]-line_len, c[1]-line_len),
#                   (c[0]+line_len, c[1]+line_len)]
#         line_2 = [(c[0]-line_len, c[1]+line_len),
#                   (c[0]+line_len, c[1]-line_len)]
#         draw = ImageDraw.Draw(image_c)
#         draw.line(line_1, fill=0, width=1)
#         draw.line(line_2, fill=0, width=1)
#         del draw
#     return image_c


def save_properties():
    '''
    Computes properties for every image in a dataset, then saves the lists
    of properties to a pickle file.
    '''
    cents_arr = []
    area_arr = []
    perim_arr = []
    data_img = np.load(os.path.join(
        'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz'))
    data_preds = np.load(PREDS_DIR)
    _ = data_img['images']
    preds = data_preds['preds']

    for pred in progress(preds):
        cents, area, perimeter = compute_properties(pred)
        cents_arr.append(cents)
        area_arr.append(area)
        perim_arr.append(perimeter)

    geometry = {
        'centroids': cents_arr,
        'areas': area_arr,
        'perimeters': perim_arr
    }
    with open(os.path.join('Output', 'Geometry', f'{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle'), 'wb') as f:
        pickle.dump(geometry, f)


def vel_norm(p1, p2, dt):
    return np.linalg.norm(np.subtract(p2, p1)*PX_TO_MM)


def time_period(ls):
    while True:
        for i in ls:
            yield i


def compute_vel():
    '''
    Compute velocity from centroid positions.
    '''
    with open('Geometry/' + DATASET + '.pickle', 'rb') as f:
        GEOMETRY = pickle.load(f)

    geom = GEOMETRY['centroids']

    vel = []
    dt = time_period([P1, P2, P2, P2])
    for i in range(len(geom)-1):
        cent1 = geom[i]
        cent2 = geom[i+1]
        if len(cent2) == 1 and len(cent1) == 1:
            vel.append(vel_norm(cent1[0], cent2[0], next(dt)))
        elif len(cent2) > 1 and len(cent1) == 1:
            v = [vel_norm(cent1[0], c, next(dt)) for c in cent2]
            vel.append(v)
        elif len(cent2) > 1 and len(cent1) > 1:
            min_idx = min_distance(cent1, cent2)
            v = [vel_norm(cent1[k[0]], cent2[k[1]], next(dt)) for k in min_idx]
            vel.append(v)
        elif len(cent2) == 1 and len(cent1) > 1:
            v = [vel_norm(c, cent2, next(dt)) for c in cent2]
            vel.append(v)

    return vel


def min_sort_row(arr):
    '''
    Receives array and orders its row by the minimum element.
    Returns row sorted array and original order.
    arr (ndarray): 2D array of numerical values.

    ex.
    arr = [[1  , 2, 2],
           [0.1, 5, 1],
           [0  , 6, 3]]
    min_sort_row(arr) -> [2, 1, 0], [[0  , 6, 3],
                                     [0.1, 5, 1],
                                     [1  , 2, 2]]
    '''
    min_d = np.amin(arr, axis=1)
    idx = np.array([i for i in range(arr.shape[0])])
    min_d_sort = min_d.argsort()
    return idx[min_d_sort], arr[min_d_sort]


def min_distance(t1, t2):
    '''
    Computes the euclidean distance between all points of t1 and t2,
    then returns which element of each list corresponds to the other
    list by minimizing distance.
    t1, t2 (List): List of tuples [(x1,y1),(x2,y2)...(xn,yn)]
    '''
    arr_s, arr_l = sorted([t1, t2], key=len)
    swap = True if arr_l == t2 else False

    distances = distance.cdist(arr_l, arr_s, 'euclidean')
    # sort rows by min value in them and keep original index order
    original_idx, distances_sorted = min_sort_row(distances)
    res = []  # saves coordinate and magnitude of min value
    exclude = set()  # prevents every point in t1/t2 to be connected to only one point in t2/t1

    for i, row in zip(original_idx, distances_sorted):
        seq = next_min(row)
        min_idx, _ = next(seq)
        if not swap:
            md = [i, min_idx]
        else:
            md = [min_idx, i]
        exclude.add(min_idx)
        res.append(md)
    if len(exclude) < len(arr_s):
        print(
            f'Not all points were connected. {len(arr_s)-len(exclude)} point(s) not used.')
    return res


def next_min(arr):
    '''
    Yields the nth index and value o minimum (idx, val).
    arr (list): List of numerical values.

    ex: arr = [0, 1, 3, 2]
        seq = mext_min(arr)
        next(seq) -> (0, 0)
        next(seq) -> (1, 1)
        next(seq) -> (3, 2)
        next(seq) -> (2, 3)
    '''
    for _ in range(len(arr)):
        yield np.argmin(arr), np.amin(arr)
        arr = np.delete(arr, np.argmin(arr))


def plot_vel(ls):
    _, (ax1, _) = plt.subplots(
        2, 1)
    for idx, v in enumerate(ls):
        try:
            ax1.plot(idx, min(v), 'rx')
        except TypeError:
            ax1.plot(idx, v, 'k.')

        # ax2.imshow(cv2.imread('HSV Frames\\Test\\' +
        #                      DATASET + '\\' + str(idx) + '.jpg', cv2.IMREAD_GRAYSCALE))
        ax1.set_ylabel('Velocity (m/s)')
        plt.pause(.0001)
    plt.show()

    # geom = GEOMETRY['centroids']
    # for i in range(len(geom)):
    #     cent = geom[i]
    #     for j in cent:
    #         if len(cent) > 1:
    #             plt.plot(j[0]*PX_TO_MM, -j[1]*PX_TO_MM, 'rx')
    #         else:
    #             plt.plot(j[0]*PX_TO_MM, -j[1]*PX_TO_MM, 'k.')
    #     plt.xlim([0, 352*PX_TO_MM])
    #     plt.ylim([-288*PX_TO_MM, 0])
    #     plt.pause(.001)
    #     plt.cla()
    # plt.show()


# def plot_pos():
#     with open('Geometry/' + DATASET + '.pickle', 'rb') as f:
#         GEOMETRY = pickle.load(f)

#     geom = GEOMETRY['centroids']
#     for idx, i in enumerate(geom):
#         if len(i) == 1:
#             plt.plot(idx, -i[0][1], 'k.')
#         elif len(i) > 1:

#             for j in i:
#                 plt.plot(idx, -j[1], 'rx')
#         # plt.xlim([0, 352*PX_TO_MM])
#         # plt.ylim([288*PX_TO_MM, 0])
#         plt.pause(.0001)
#         # plt.cla()
#     plt.show()


# def number_of_droplets():
#     geom = GEOMETRY['centroids']
#     n_droplets = [len(g) for g in geom]
#     return n_droplets


# %% MAIN
if __name__ == "__main__":
    save_properties()
