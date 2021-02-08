'''
Test previously trained model and save predictions.
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from progressbar import progressbar as progress
from utils import losses
from utils.preprocessing import normalizeuint8
from utils.misc import chunks, get_concat_h


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# %% VARIABLES

ARCHITECTURE_NAME = 'multires'
DATASET = 'Globular'
N_FILTERS = 16
BATCH_SIZE_TRAIN = 16
EPOCHS = 100
MODEL_DIR = os.path.join('Output', 'Saved Models',
                         f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}')
BATCH_SIZE = 1000

# %% FUNCTIONS


def test_model(model_dir, batch_size=None):
    '''
    Tests trained model and returns masks and loss.
    '''
    try:
        with open(model_dir + '/trainHistoryDict', 'rb') as file_pi:
            params = pickle.load(file_pi)['params']
    except FileNotFoundError:
        print(
            f"Oops! It seems the model folder '~/{MODEL_DIR}' does not exist.")
        return
    model = tf.keras.models.load_model(params['save_dir'], compile=False)

    if params['optimizer_name'] == 'adam':
        opt = tf.optimizers.Adam(params['lr'])

    if params['loss_name'] == 'iou':
        loss_fn = losses.iou_coef

    model.compile(
        optimizer=opt, loss=loss_fn)

    data = np.load(os.path.join('Data', 'Image', 'Input',
                                f"{params['dataset'].lower()}_gray.npz"))
    if batch_size:
        test_size = len(data['images'])
    else:
        test_size = 10
        batch_size = 10
    id_batches = chunks([i for i in range(test_size)], batch_size)
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
    predictions = np.array(predictions, dtype=np.uint8)
    np.savez(os.path.join('Output', 'Predictions',
                          f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds'), preds=predictions)
    print(f'Mean loss - {sum(results)/len(results):.2f}')


def plot_preds(dataset):
    data_img = np.load(os.path.join(
        'Data', 'Image', 'Input', f'{dataset.lower()}_rgb.npz'))

    data_pred = np.load(os.path.join('Output', 'Predictions',
                                     f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}_preds.npz'))

    images = data_img['images']
    preds = data_pred['preds']
    _, ax = plt.subplots(1, 2)
    for i, p in zip(images, preds):
        ax[0].imshow(i)
        ax[1].imshow(p)
        plt.pause(.005)
    plt.show()


def main():
    '''
    Main.
    '''
    test_model(MODEL_DIR, BATCH_SIZE)


# %%MAIN


if __name__ == "__main__":
    main()
