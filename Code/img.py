import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import properties

DATASET = 'Globular'
num = 500
pos, area, perimeter = properties.compute_centroids(num, DATASET)
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

fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.imshow(image)
ax1.plot(strip, 'k', linewidth=1)
ax1.plot(mask_strip, 'r', linewidth=1)
ax1.plot([0, 351], [h, h], linewidth=1)


ax2.plot(strip, linewidth=1)
ax2.plot(mask_strip, linewidth=1)
ax2.set_xlim([0, 351])

ax3.imshow(mask)
plt.show()
