'''
Test previously trained model and save predictions.
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import progressbar as progress
from utils import losses
from utils.misc import chunks, set_size
from utils.preprocessing import normalizeuint8

# video settings
matplotlib.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
# output images for LaTex
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
# set style
sns.set()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(error)
# %% VARIABLES
# Pixel to distance conversions
# 26 px = .045 in = 1.143 mm
# 1px = 0.04396153846153846 mm = 4.396153846153846 * 10^(-5) m
#
# LaTex \textwidth = 472.03123 pt. Useful for figure sizing

ARCHITECTURE_NAME = 'unet'
DATASET = 'Spray'
N_FILTERS = 8
BATCH_SIZE_TRAIN = 8

EPOCHS = 200
MODEL_DIR = os.path.join('Output', 'Saved Models',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}')
BATCH_SIZE = 1

# %% FUNCTIONS


def test_model():
    '''
    Tests trained model and returns masks and loss.
    '''
    try:
        with open(os.path.join(MODEL_DIR, 'trainHistoryDict'), 'rb') as file_pi:
            params = pickle.load(file_pi)['params']
    except FileNotFoundError:
        print(
            f"Oops! It seems the model folder '~/{MODEL_DIR}' does not exist.")
        return
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)

    if params['optimizer_name'] == 'adam':
        opt = tf.optimizers.Adam(params['learning_rate'])

    if params['loss_name'] == 'iou':
        loss_fn = losses.jaccard_distance_loss

    model.compile(
        optimizer=opt, loss=loss_fn)

    data = np.load(os.path.join('Data', 'Image', 'Input',
                                f"{params['dataset'].lower()}_gray.npz"))

    test_size = len(data['images'])
    id_batches = chunks([i for i in range(test_size)], BATCH_SIZE)
    results = []

    images = data['images'].astype('float32')
    images = images/255
    predictions = []
    for batch in progress(id_batches):
        print(f'Testing batch [{batch[0]} - {batch[-1]}]')
        x_batch = images[batch]
        y_batch = model.predict(x_batch, verbose=0)
        loss = model.evaluate(x_batch, y_batch, verbose=0)
        for pred in y_batch:
            predictions.append(normalizeuint8(pred[..., 0]))
        results.append(loss)
    test_loss = {'test_loss': results}
    with open(os.path.join(MODEL_DIR, 'test_loss_dict'), 'wb') as file_pi:
        pickle.dump(test_loss, file_pi)

    predictions = np.array(predictions, dtype=np.uint8)
    np.savez(os.path.join('Output', 'Predictions',
                          f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds'), preds=predictions)
    print(f'Mean loss - {sum(results)/len(results):.2f}')


def plot_preds():
    '''
    DOC
    '''
    data_img = np.load(os.path.join(
        'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz'))

    data_pred = np.load(os.path.join('Output', 'Predictions',
                                     f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz'))

    images = data_img['images']
    preds = data_pred['preds']

    with sns.axes_style('dark'):
        _, axes = plt.subplots(1, 2)

    for i, pred in zip(images, preds):
        axes[0].imshow(i)
        axes[1].imshow(pred, cmap='gray')
        plt.pause(.005)
    plt.show()


def plot_loss_samples():
    '''
    DOC
    '''
    with open(os.path.join(MODEL_DIR, 'test_loss_dict'), 'rb') as file_pi:
        data = pickle.load(file_pi)
    test_loss = data['test_loss']
    fig, axes = plt.subplots(1, 1, figsize=set_size(
        472.03123, 0.5, aspect_ratio=0.9))
    hist, bins, patches = axes.hist(test_loss, log=True)
    for rectangle in patches:
        rectangle.set_y(0.1)
        rectangle.set_height(rectangle.get_height() - 0.1)
    samples = []
    for i in range(len(bins)-1):
        try:
            samples.append(np.random.choice(np.where((bins[i] <= test_loss) & (
                test_loss <= bins[i+1]))[0]))
        except ValueError:
            continue
    axes.set_xlabel('Test loss')
    axes.set_ylabel('Frequency')
    fig.tight_layout()
    os.makedirs(os.path.join('Output', 'Plots',
                             f'{DATASET.lower()}_test_loss'), exist_ok=True)
    plt.show()
    # fig.savefig(os.path.join('Output', 'Plots',
    #                          f'{DATASET.lower()}_test_loss', f'{DATASET.lower()}_test_loss.pgf'), bbox_inches='tight')
    data_img = np.load(os.path.join(
        'Data', 'Image', 'Input', f'{DATASET.lower()}_rgb.npz'))

    data_pred = np.load(os.path.join('Output', 'Predictions',
                                     f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz'))

    images = data_img['images'][samples]
    preds = data_pred['preds'][samples]

    for i, img_pred in enumerate(zip(images, preds)):
        img, pred = img_pred
        with sns.axes_style('dark'):
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, frameon=False, figsize=set_size(
                472.03123, 0.5, aspect_ratio=img.shape[0]/img.shape[1], subplots=(1, 2)))
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[0].imshow(img)
        axes[0].set_title(f'Frame {samples[i]}')
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title(f'Test loss = {test_loss[samples[i]]:.4f}')

        os.makedirs(os.path.join('Output', 'Plots',
                                 f'{DATASET.lower()}_test_loss', f'{DATASET.lower()}_test_loss_{i+1}'), exist_ok=True)
        fig.savefig(os.path.join('Output', 'Plots', f'{DATASET.lower()}_test_loss', f'{DATASET.lower()}_test_loss_{i+1}',
                                 f'{DATASET.lower()}_test_loss_{i+1}.pgf'), bbox_inches='tight')


def main():
    '''
    Main.
    '''
    test_model()


# %%MAIN


if __name__ == "__main__":
    plot_loss_samples()
