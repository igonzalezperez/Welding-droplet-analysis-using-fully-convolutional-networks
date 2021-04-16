'''
Train UNET model and save results
'''
# %% IMPORTS
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from progressbar import progressbar as progress
from architectures import UNET, DECONVNET, MULTIRES
from utils.postprocessing import set_size, latex_plot_config

# video settings
matplotlib.rcParams['animation.ffmpeg_path'] = os.path.abspath(
    'C:\\ffmpeg\\bin\\ffmpeg.exe')
# set style
sns.set_style('whitegrid')

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

EPOCHS = 200
LOSS_NAME = 'iou'
BATCH_SIZE = 8
N_FILTERS = 8
LEARNING_RATE = .001
OPTIMIZER_NAME = 'adam'
ARCHITECTURE_NAME = 'unet'

DATASET = 'Spray'

SAVE_DIR = os.path.join('Output', 'Saved Models',
                        f'{ARCHITECTURE_NAME.lower()}_{DATASET.lower()}_{N_FILTERS}_{BATCH_SIZE}_{EPOCHS}')
CHECKPOINT_DIR = os.path.join(
    'Output', 'Saved Models', 'Checkpoints', '{epoch:02d}-{val_loss:.2f}')

PARAMS = {'dataset': DATASET,
          'epochs': EPOCHS,
          'loss_name': LOSS_NAME,
          'batch_size': BATCH_SIZE,
          'n_filters': N_FILTERS,
          'learning_rate': LEARNING_RATE,
          'optimizer_name': OPTIMIZER_NAME,
          'architecture_name': ARCHITECTURE_NAME,
          'save_dir': SAVE_DIR,
          'checkpoint_dir': CHECKPOINT_DIR
          }

# %%FUNCTIONS


def choose_architecture(arch_name):
    '''
    DOC
    '''
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


def train_model(params, save=False, verbose=2, gridsearch=False, folds=5):
    '''
    Trains new UNET model and saves results.
    '''
    images, masks, shape = load_dataset(
        params['dataset'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        params['checkpoint_dir'], monitor='val_loss', save_best_only=True, save_weights_only=True)
    if gridsearch:
        kfold = KFold(folds, shuffle=True)
        cross_val = {'epoch': [], 'loss': [], 'val_loss': []}
        model_arch = choose_architecture(params['architecture_name'])
        for train_idx, val_idx in progress(kfold.split(images)):
            model = model_arch(n_filters=params['n_filters'],
                               input_shape=shape,
                               optimizer_name=params['optimizer_name'],
                               learning_rate=params['learning_rate'],
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
            cross_val['loss'].append(history['loss'][-1])
            cross_val['val_loss'].append(history['val_loss'][-1])
            cross_val['epoch'].append(len(history['loss']))
        return cross_val
    else:
        model_arch = choose_architecture(params['architecture_name'])
        model = model_arch(n_filters=params['n_filters'],
                           input_shape=shape,
                           optimizer_name=params['optimizer_name'],
                           learning_rate=params['learning_rate'],
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
                         callbacks=[early_stop, checkpoint])
        history = hist.history
        history['params'] = params
        if save:
            model.load_weights(tf.train.latest_checkpoint(
                os.path.dirname(params['checkpoint_dir'])))
            model.save(params['save_dir'])
            with open(os.path.join(params['save_dir'], 'trainHistoryDict'), 'wb') as file_pi:
                pickle.dump(history, file_pi)

            plot_train_loss(params)
    return history


def plot_train_loss(params):
    '''
    Plot training loss curves of trained model.
    '''
    with open(os.path.join(params['save_dir'], 'trainHistoryDict'), 'rb') as file_pi:
        history = pickle.load(file_pi)
    fig, axes = plt.subplots(1, 1, figsize=set_size(
        472.03123, 1, aspect_ratio=.4))
    axes.plot(history['loss'], label='Train')
    axes.plot(history['val_loss'], label='Validation')
    axes.axvline(x=history['val_loss'].index(
        min(history['val_loss']))+1, color='k', linestyle='--', label='Best model')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.legend()
    fig.tight_layout()
    fig.savefig(os.path.join('Output', 'Plots',
                             f'{params["dataset"].lower()}_train_loss_curve.png'), bbox_inches='tight', transparent=True, dpi=300)
    latex_plot_config()
    fig.savefig(os.path.join('Output', 'Plots',
                             f'{params["dataset"].lower()}_train_loss_curve.pgf'), bbox_inches='tight')


def main():
    '''
    Train model
    '''

    train_model(PARAMS, verbose=1, save=True)
# %%MAIN


if __name__ == "__main__":
    plot_train_loss(PARAMS)
