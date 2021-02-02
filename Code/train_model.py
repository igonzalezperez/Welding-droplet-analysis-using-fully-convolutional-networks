'''
Train UNET model and save results
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from progressbar import progressbar as progress
from architectures import UNET, DECONVNET, MULTIRES

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# %% VARIABLES
EPOCHS = 100
LOSS_NAME = 'iou'
BATCH_SIZE = 16
N_FILTERS = 16
LEARNING_RATE = .001
OPTIMIZER_NAME = 'adam'
ARCHITECTURE_NAME = 'multires'

DATASET = 'Globular'

TRAIN = True

SAVE_DIR = os.path.join('Output', 'Saved Models',
                        f'{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE}_{EPOCHS}')

PARAMS = {'dataset': DATASET,
          'epochs': EPOCHS,
          'loss_name': LOSS_NAME,
          'batch_size': BATCH_SIZE,
          'n_filters': N_FILTERS,
          'lr': LEARNING_RATE,
          'optimizer_name': OPTIMIZER_NAME,
          'architecture_name': ARCHITECTURE_NAME,
          'save_dir': SAVE_DIR
          }

# %%FUNCTIONS


def choose_architecture(arch_name):
    if arch_name == 'unet':
        return UNET
    elif arch_name == 'deconvnet':
        return DECONVNET
    elif arch_name == 'multires':
        return MULTIRES


def load_dataset(dataset):
    '''
    Loads dataset and splits into train and validation.
    '''
    data = np.load(os.path.join('Data', 'Image', 'Augmented',
                                dataset.lower() + '_augmented.npz'))
    images = data['images']
    images = np.expand_dims(images, axis=-1).astype('float32')
    images = images/255

    masks = data['masks']
    masks = np.expand_dims(masks, axis=-1).astype('float32')
    masks = masks/255

    shape = images.shape[1:]
    return images, masks, shape


def train_model(params, save=False, verbose=0, gridsearch=False, folds=5):
    '''
    Trains new UNET model and saves results.
    '''
    images, masks, shape = load_dataset(
        params['dataset'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=100)
    if gridsearch:
        kfold = KFold(folds, shuffle=True)
        cv = {'epoch': [], 'loss': [], 'val_loss': []}
        model_arch = choose_architecture(params['architecture_name'])
        for train_idx, val_idx in progress(kfold.split(images)):
            model = model_arch(n_filters=params['n_filters'],
                               input_shape=shape,
                               optimizer_name=params['optimizer_name'],
                               lr=params['lr'],
                               loss_name=params['loss_name']).create_model()
            x_train = images[train_idx]
            y_train = masks[train_idx]

            x_val = images[val_idx]
            y_val = masks[val_idx]

            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).repeat().batch(params['batch_size'])
            val_ds = tf.data.Dataset.from_tensor_slices(
                (x_val, y_val)).repeat().batch(params['batch_size'])
            hist = model.fit(train_ds, epochs=params['epochs'],
                             steps_per_epoch=x_train.shape[0]//params['batch_size'],
                             validation_data=val_ds,
                             validation_steps=x_val.shape[0]//params['batch_size'], verbose=verbose,
                             callbacks=[early_stop])
            history = hist.history
            cv['loss'].append(history['loss'][-1])
            cv['val_loss'].append(history['val_loss'][-1])
            cv['epoch'].append(len(history['loss']))
            return cv
    else:
        model_arch = choose_architecture(params['architecture_name'])
        model = model_arch(n_filters=params['n_filters'],
                           input_shape=shape,
                           optimizer_name=params['optimizer_name'],
                           lr=params['lr'],
                           loss_name=params['loss_name']).create_model()
        x_train, x_val, y_train, y_val = train_test_split(
            images, masks, test_size=0.2)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).repeat().batch(params['batch_size'])
        val_ds = tf.data.Dataset.from_tensor_slices(
            (x_val, y_val)).repeat().batch(params['batch_size'])
        hist = model.fit(train_ds, epochs=params['epochs'],
                         steps_per_epoch=x_train.shape[0]//params['batch_size'],
                         validation_data=val_ds,
                         validation_steps=x_val.shape[0]//params['batch_size'], verbose=verbose,
                         callbacks=[early_stop])
        history = hist.history
        history['params'] = params
        if save:
            model.save(params['save_dir'])
            with open(params['save_dir'] + '/trainHistoryDict', 'wb') as file_pi:
                pickle.dump(history, file_pi)

            plt.plot(history['loss'], label='Loss')
            plt.plot(history['val_loss'], label='Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(params['save_dir'] + '/loss_curve', format='pdf')
    return history


def main():
    '''
    Train model
    '''

    train_model(PARAMS, verbose=1, save=True)
# %%MAIN


if __name__ == "__main__":
    main()
