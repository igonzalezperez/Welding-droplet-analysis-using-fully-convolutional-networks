import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from compute_properties import compute_properties

DATASET = 'Globular'
num = 500
pos, area, perimeter = compute_properties(num, DATASET)
m = np.argmax(area)
h = pos[m][1]
area = area[m]
perimeter = perimeter[m]

image = Image.open(
    os.getcwd() + '\\HSV Frames\\Test\\' + DATASET + '\\'+str(num) + '.jpg').convert('L')

mask = Image.open(
    os.getcwd() + '\\HSV Frames\\Preds\\'+DATASET+'\\Masks\\'+str(num) + '.jpg')

image = np.asarray(image)
mask = np.asarray(mask)

strip = image[h, :]
mask_strip = mask[h, :]

fig1 = plt.figure()
plt.imshow(image)
plt.plot(strip, 'k', linewidth=1)
plt.plot(mask_strip, 'r', linewidth=1)
plt.plot([0, strip.shape[0]-2], [h, h], linewidth=1)
plt.tight_layout()
plt.savefig(f'boundary_{DATASET}_{num}_1.pdf', format='pdf')

fig2 = plt.figure()
plt.plot(strip, linewidth=1, label='real pixel value')
plt.plot(mask_strip, linewidth=1, label='prediction')
plt.xlim([0, strip.shape[0]-2])
plt.legend()
plt.tight_layout()
plt.savefig(f'boundary_{DATASET}_{num}_2.pdf', format='pdf')

fig3 = plt.figure()
plt.imshow(mask)
plt.tight_layout()
plt.savefig(f'boundary_{DATASET}_{num}_3.pdf', format='pdf')
plt.show()
