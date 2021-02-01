'''
Test previously trained model and save predictions.
'''
# %% IMPORTS
from utils.utils import chunks, get_concat_h
from utils import losses
from PIL import Image
import numpy as np
import os
import pickle
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


ARCHITECTURE_NAME = 'unet'
DATASET = 'Spray'
N_FILTERS = 16
BATCH_SIZE_TRAIN = 16
EPOCHS = 100
MODEL_DIR = f'Saved Models/{ARCHITECTURE_NAME}_{DATASET}_{N_FILTERS}_{BATCH_SIZE_TRAIN}_{EPOCHS}'
BATCH_SIZE = 1000


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
    elif params['optimizer_name'] == 'adadelta':
        opt = tf.optimizers.Adadelta(params['lr'])

    if params['loss_name'] == 'iou':
        loss_fn = losses.iou_coef

    model.compile(
        optimizer=opt, loss=loss_fn)
    if batch_size:
        test_size = len(os.listdir('HSV Frames/Test/' + params['dataset']))
    else:
        test_size = 10
        batch_size = 10
    id_batches = chunks([i for i in range(test_size)], batch_size)
    results = []
    for batch in id_batches:
        images_test = []
        for i in batch:
            img = Image.open('HSV Frames/Test/' +
                             params['dataset'] + '/' + str(i) + '.jpg').convert('L')
            images_test.append(np.asarray(img))
        images_test = np.array(images_test).astype('float32')
        x_test = images_test/255

        print(f'Testing batch [{batch[0]} - {batch[-1]}]')
        y_test = model.predict(x_test, verbose=0)
        loss = model.evaluate(x_test, y_test, verbose=0)
        results.append(loss)
        print(f'Batch loss - {loss:.2f}')
        for i, j in enumerate(batch):
            img = Image.open('HSV Frames/Test/' +
                             params['dataset'] + '/' + str(j) + '.jpg')
            img_mask = Image.fromarray(
                y_test[i][:, :, 0]*255).convert('L')
            img_mask.save(
                'HSV Frames/Preds/'+DATASET+'/Masks/' + str(j) + '.jpg')
            get_concat_h(img, img_mask).save(
                'HSV Frames/Preds/'+DATASET+'/Inputs and Masks/' + str(j) + '.jpg')
    print(f'Mean loss - {sum(results)/len(results):.2f}')


def main():
    '''
    Main.
    '''
    test_model(MODEL_DIR, BATCH_SIZE)

# %%MAIN


if __name__ == "__main__":
    main()
