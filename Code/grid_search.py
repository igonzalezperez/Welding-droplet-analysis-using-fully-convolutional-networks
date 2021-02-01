'''
Grid. search
'''
# %%
import os
import pickle
import itertools
import pandas as pd
import tensorflow as tf
from train_model import train_model
from utils import losses
from architectures import UNET, DECONVNET, MULTIRES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# %%
DATASET = ['Spray']
EPOCHS = [100]
LOSS_NAME = ['iou']
BATCH_SIZE = [8, 16, 32, 64]
N_FILTERS = [8, 16, 32, 64]
LEARNING_RATE = [0.01, 0.007, 0.003, .001]
OPTIMIZER_NAME = ['adam', 'adadelta']
ARCHITECTURE_NAME = ['unet', 'deconvnet', 'multires']

PARAMS = {'dataset': DATASET,
          'epochs': EPOCHS,
          'loss_name': LOSS_NAME,
          'batch_size': BATCH_SIZE,
          'n_filters': N_FILTERS,
          'lr': LEARNING_RATE,
          'optimizer_name': OPTIMIZER_NAME,
          'architecture_name': ARCHITECTURE_NAME
          }


def get_grid(params):
    '''
    Receives dict of parameters and returns list of dicts with every possible combination.
    e.g. {'filters':[1,2], 'epochs':[5,6]} -> [{'filters':1, 'epoch':5},{'filters':1, 'epoch':6},
    {'filters':2, 'epoch':5}, {'filters':2, 'epoch':6}].
    '''
    grid_space = []
    keys = list(params)
    for values in itertools.product(*map(params.get, keys)):
        grid_space.append(dict(zip(keys, values)))
    return grid_space
# %%


def grid_search(params):
    '''
    Perform grid search over filters, batch size and epochs. Both the model and model's history are
    saved.
    '''
    grid_space = get_grid(params)
    with open('grid.pkl', 'wb') as f:
        pickle.dump(grid_space, f, pickle.HIGHEST_PROTOCOL)
    grid_frame = pd.DataFrame(grid_space)

    print(f'Grid searching {len(grid_space)} different sets of parameters')

    records = []

    for i, p in enumerate(grid_space):
        print(f'Training model {i+1}: {p}')
        cv = train_model(p, save=False, gridsearch=True, verbose=0, folds=4)
        epoch_dict = {'epoch_'+str(i): v for i, v in enumerate(cv['epoch'])}
        loss_dict = {'loss_'+str(i): v for i, v in enumerate(cv['loss'])}
        val_loss_dict = {'val_loss_' +
                         str(i): v for i, v in enumerate(cv['val_loss'])}
        r = {**p, **epoch_dict, **loss_dict, **val_loss_dict}

        records.append(r)
        loss = sum(cv['loss'])/len(cv['loss'])
        val_loss = sum(cv['val_loss'])/len(cv['val_loss'])
        print(f'''loss={loss:.4f}''')
        print(f'''val_loss={val_loss:.4f}\n''')
        records_frame = pd.DataFrame(records)
        filename = f"Records/{r['dataset']}_records_rerun.xlsx"
        records_frame.to_excel(filename)

    return records


def main():
    '''
    Main.
    '''
    grid_search(PARAMS)


if __name__ == "__main__":
    main()
