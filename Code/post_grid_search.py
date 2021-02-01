import pickle
import numpy as np
with open('records.pkl', 'rb') as f:
    records = pickle.load(f)
for i, x in enumerate(records):
    print(x['loss'])
    print(x['val_loss'])
    loss = np.mean(x['loss'])
    loss_std = np.std(x['loss'])

    val_loss = np.mean(x['val_loss'])
    val_loss_std = np.std(x['val_loss'])

    print(f'mean loss: {loss:.4f}')
    print(f'std loss: {loss_std:.4f}\n')

    print(f'mean val loss: {val_loss:.4f}')
    print(f'mean val loss std: {val_loss:.4f}\n\n')

##
# ARCHITECTURE = [UNET, DECONVNET, MULTIRES]
# DATASET = ['Globular']
# SHAPE = [(352, 296, 1)]
# N_FILTERS = [4, 8, 16, 32]
# BATCH_SIZE = [8, 16, 32]
# EPOCHS = [100]
# LEARNING_RATE = [0.001, .005, .01]
# OPTIMIZER = [tf.optimizers.Adam, tf.optimizers.Adadelta]
# LOSS_FN = [losses.iou_coef]
