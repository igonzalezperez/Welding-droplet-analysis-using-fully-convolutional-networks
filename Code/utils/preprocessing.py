'''Convert .jpg images to smaller size and grayscale and save in npz file'''
# %% Imports
import os
import numpy as np
import progressbar
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import load_images_paths, preprocess, ProjectPaths
# %% Reshape and B&W images
PATHS = ProjectPaths()
IM_WIDTH = 112  # has to be multiple of 4
IM_HEIGHT = 96  # has to be multiple of 4
ID = 0  # choose video
IMAGE_PATHS = load_images_paths(PATHS.jpg_frames_paths[ID])
N_IMAGES = len(IMAGE_PATHS)


def show_batch(image_batch):
    '''Plot 25 image examples.'''
    image_batch = (image_batch*255).round().astype(np.uint8)
    fig, _ = plt.subplots(figsize=(10, 10))
    for num in range(25):
        plt.subplot(5, 5, num+1)
        plt.imshow(np.squeeze(image_batch[num]))
        plt.axis('off')
    fig.suptitle(
        f'Training set sample images\n{os.path.basename(PATHS.jpg_frames_paths[ID])}')
    plt.show()


def save_npz():
    '''Save array of images as npz file.'''
    data = []
    for i in progressbar.progressbar(range(N_IMAGES)):
        image = preprocess(IMAGE_PATHS[i], resize_shape=(
            IM_WIDTH, IM_HEIGHT), grayscale=False)
        data.append(image)

    data = np.asarray(data, dtype='float32')/255.
    np.savez(PATHS.npz_dataset_paths[ID] + '_ordered',
             data=data)
    idx = np.random.randint(len(data) - 25)
    image_batch = data[idx:idx+25]
    show_batch(image_batch)
    np.random.shuffle(data)
    idx = np.random.randint(len(data) - 25)
    image_batch = data[idx:idx+25]
    show_batch(image_batch)
    print(f'Data: {data.shape}')

    x_train, x_test = train_test_split(data, test_size=.2, random_state=42)
    np.savez(PATHS.npz_dataset_paths[ID],
             x_train=x_train, x_test=x_test)
    #  data_train, data_test = train_test_split(
    #     data, test_size=.2, random_state=42)
    # idx = np.random.randint(len(data_train)-25)
    # image_batch = data_train[idx:idx+25]
    # show_batch(image_batch)
    # print(f'Train data: {data_train.shape}')
    # print(f'Test data: {data_test.shape}')
    # np.savez(PATHS.npz_dataset_paths[ID],
    #          images_train=data_train.astype('float32'), images_test=data_test.astype('float32'))


def save_test():
    '''
    Doc
    '''
    for i, folder in enumerate(PATHS.jpg_frames_paths):
        if i == 0:
            save_file = 'HSV Frames/Test/Globular/'
            shape = (352, 296)
        elif i == 1:
            save_file = 'HSV Frames/Test/Spray/'
            shape = (352, 352)
        for name in os.listdir(folder):
            img = Image.open(folder + '/' + name)
            im_rs = img.resize(shape)
            im_rs.save(save_file + name)


# %%
if __name__ == "__main__":
    save_test()
