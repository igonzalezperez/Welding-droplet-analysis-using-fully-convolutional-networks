'''
Compute mask properties
'''
# %% IMPORTS
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar as progress
from scipy.spatial import distance
from cv2 import cv2
from utils.postprocessing import time_seq

# %% VARIABLES
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
PX_TO_MM = 1.143/26

ARCHITECTURE_NAME = 'unet'
DATASET = 'Spray'
N_FILTERS = 8
BATCH_SIZE_TRAIN = 8
EPOCHS = 200
PREDS_DIR = os.path.join('Output', 'Predictions',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz')

# %% FUNCTIONS


def compute_properties(img):
    '''
    Returns centroid, perimeter and area for every contour in an image as a list.
    img (ndarray): Image segmentation map.
    '''
    _, thresh = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids_float = []
    centroids = []
    area = []
    perimeter = []
    volume = []
    volume_corrected = []
    for cnt in contours:
        mmt = cv2.moments(cnt)
        if mmt['m00'] > 20:  # ignore really small contours
            try:
                c_x = mmt['m10']/mmt['m00']
                c_y = mmt['m01']/mmt['m00']
                centroids_float.append((c_x, c_y))
                c_x = int(c_x)
                c_y = int(c_y)
                centroids.append((c_x, c_y))

                area.append(mmt['m00']*PX_TO_MM**2)
                perimeter.append(cv2.arcLength(cnt, True)*PX_TO_MM*0.86)

                ellipse = cv2.fitEllipse(cnt)
                semi_major = max(ellipse[1])/2
                semi_minor = min(ellipse[1])/2
                vol = (4/3)*np.pi*semi_minor**2*semi_major
                volume.append(vol*PX_TO_MM**3)
                volume_corrected.append(vol*PX_TO_MM**3*0.86**3)
                # plt.imshow(thresh)
                # plt.show()
                # x = ellipse[0][0]
                # y = ellipse[0][1]
                # a = ellipse[1][0]/2
                # b = ellipse[1][1]/2
                # alpha = ellipse[2]*np.pi/180
                # x_a = x + a*np.cos(alpha)
                # y_a = y + a*np.sin(alpha)
                # x_b = x + b*np.cos(np.pi/2+alpha)
                # y_b = y + b*np.sin(np.pi/2+alpha)
                # plt.plot(x, y, 'rx')
                # plt.plot(x_a, y_a, 'rx')
                # plt.plot(x_b, y_b, 'rx')
                # plt.imshow(thresh)
                # thresh = cv2.ellipse(
                #     np.zeros((thresh.shape[0], thresh.shape[1], 3)), ellipse, (255, 255, 255), 2)
                # plt.imshow(thresh)
                # plt.show()
                # breakpoint()
            except ZeroDivisionError:
                continue
    return centroids_float, centroids, area, perimeter, volume, volume_corrected


def save_properties():
    '''
    Computes properties for every image in a dataset, then saves the lists
    of properties to a pickle file.
    '''
    cents_float_arr = []
    cents_arr = []
    area_arr = []
    perim_arr = []
    vol_arr = []
    vol_corrected_arr = []
    time_list = []
    time_cycle = time_seq()

    data_img = np.load(os.path.join(
        'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz'))
    data_preds = np.load(PREDS_DIR)
    _ = data_img['images']
    preds = data_preds['preds']

    for _, pred in progress(enumerate(preds)):
        try:
            last_time = time_list[-1]
            time_list.append(last_time + next(time_cycle))
        except IndexError:
            time_list.append(0)
        cents_float, cents, area, perimeter, volume, volume_corrected = compute_properties(
            pred)
        cents_float_arr.append(cents_float)
        cents_arr.append(cents)
        area_arr.append(area)
        perim_arr.append(perimeter)
        vol_arr.append(volume)
        vol_corrected_arr.append(volume_corrected)
    geometry = {
        'centroids_float': cents_float_arr,
        'centroids': cents_arr,
        'areas': area_arr,
        'perimeters': perim_arr,
        'volumes': vol_arr,
        'volumes_corrected': vol_corrected_arr,
        'time': time_list
    }
    with open(os.path.join('Output', 'Geometry', f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_geometry.pickle'), 'wb') as data_file:
        pickle.dump(geometry, data_file)


def vel_norm(p_1, p_2):
    '''
    Calculates the norm of a vector that goes from p_1 to p_2
    '''
    return np.linalg.norm(np.subtract(p_2, p_1))


def list_loop(input_list):
    '''
    Yields every element of a list in a cycle. e.g. [1, 2, 3] will yield ->1, 2, 3, 1, 2...
    '''
    while True:
        for i in input_list:
            yield i


def compute_vel():
    '''
    Compute velocity from centroid positions.
    '''
    with open('Geometry/' + DATASET + '.pickle', 'rb') as data_file:
        geometry = pickle.load(data_file)

    geom = geometry['centroids']

    velocity = []
    # dt = time_period([P1, P2, P2, P2])
    for i in range(len(geom)-1):
        cent1 = geom[i]
        cent2 = geom[i+1]
        if len(cent2) == 1 and len(cent1) == 1:
            velocity.append(vel_norm(cent1[0], cent2[0]))
        elif len(cent2) > 1 and len(cent1) == 1:
            vel = [vel_norm(cent1[0], c) for c in cent2]
            velocity.append(vel)
        elif len(cent2) > 1 and len(cent1) > 1:
            min_idx = min_distance(cent1, cent2)
            vel = [vel_norm(cent1[k[0]], cent2[k[1]]) for k in min_idx]
            velocity.append(vel)
        elif len(cent2) == 1 and len(cent1) > 1:
            vel = [vel_norm(c, cent2) for c in cent2]
            velocity.append(vel)

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


def min_distance(t_1, t_2):
    '''
    Computes the euclidean distance between all points of t1 and t2,
    then returns which element of each list corresponds to the other
    list by minimizing distance.
    t1, t2 (List): List of tuples [(x1,y1),(x2,y2)...(xn,yn)]
    '''
    arr_s, arr_l = sorted([t_1, t_2], key=len)
    swap = True if arr_l == t_2 else False

    distances = distance.cdist(arr_l, arr_s, 'euclidean')
    # sort rows by min value in them and keep original index order
    original_idx, distances_sorted = min_sort_row(distances)
    res = []  # saves coordinate and magnitude of min value
    exclude = set()  # prevents every point in t1/t2 to be connected to only one point in t2/t1

    for i, row in zip(original_idx, distances_sorted):
        seq = next_min(row)
        min_idx, _ = next(seq)
        if not swap:
            min_coords = [i, min_idx]
        else:
            min_coords = [min_idx, i]
        exclude.add(min_idx)
        res.append(min_coords)
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


# %% MAIN
if __name__ == "__main__":
    save_properties()
