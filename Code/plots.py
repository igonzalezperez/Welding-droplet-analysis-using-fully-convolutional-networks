import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

model = 'Globular_8_8_100'
hist = pd.read_pickle(os.getcwd()+'\\Saved Models\\' +
                      model + '\\trainHistoryDict')
loss = hist['loss']
val_loss = hist['val_loss']

plt.plot(loss, label='Loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.getcwd()+'\\Saved Models\\' + model + '\\loss_curve.pdf')
plt.show()
