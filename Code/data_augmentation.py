'''
Augment data
'''
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar as progress
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


# %% VARIABLES
DATASET = ('Globular', 'Spray')
N_AUGMENT = 5

if DATASET == 'Spray':
    IMG_WIDTH = 352
    IMG_HEIGHT = 352
elif DATASET == 'Globular':
    IMG_WIDTH = 352
    IMG_HEIGHT = 296

# Define our augmentation pipeline.
SEQ = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    # rotate by -45 to 45 degrees (affects segmaps)
    iaa.Affine(rotate=(-45, 45)),
    # apply water effect (affects segmaps)
    iaa.ElasticTransformation(alpha=50, sigma=5)
], random_order=True)


def augment_data(seq, img, segmap, n_images):
    '''
    Recieves images and masks and returns augmented versions.
    '''
    images_aug = []
    segmaps_aug = []
    for _ in range(n_images):
        images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)
    return images_aug, segmaps_aug


def plot_augmented_samples(dataset):
    def get_coords(n_images=12, n_rows=3):
        coords = []
        n_cols = int(np.ceil(n_images/n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                coords.append((i, j))
        for p in coords:
            yield p

    coord = get_coords()
    data = np.load(os.path.join('Data', 'Image', 'Labelbox',
                                dataset.lower() + '_segmented.npz'))

    n = np.random.randint(low=0, high=len(data['images']))
    img = data['images'][n]
    mask = data['masks'][n]
    _, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(10, 10))
    c1 = next(coord)
    c2 = next(coord)
    ax[c1].imshow(img)
    ax[c2].imshow(mask)
    ax[c1].set_title('Original image')
    ax[c2].set_title('Original mask')

    mask = mask.astype(bool)
    img_shape = img.shape

    segmap = SegmentationMapsOnImage(mask, shape=img_shape)
    img_aug, mask_aug = augment_data(SEQ, img, segmap, 5)

    for i, m in zip(img_aug, mask_aug):
        sg_map = m.draw(size=img_shape)[0]
        sg_map = ((sg_map[..., 0] != 0)*255).astype(np.uint8)

        ax[next(coord)].imshow(i)
        ax[next(coord)].imshow(sg_map)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def main():
    '''
    Load original images and masks, augment them and save as images.
    '''
    for d in DATASET:
        data = np.load(os.path.join('Data', 'Image', 'Labelbox',
                                    d.lower() + '_segmented.npz'))
        images = data['images']
        masks = data['masks'].astype(bool)
        image_shape = images.shape[1:3]

        images_augmented, masks_augmented = ([], [])
        for image, mask in progress(zip(images, masks)):
            segmap = SegmentationMapsOnImage(mask, shape=image_shape)
            image_aug, segmap_aug = augment_data(SEQ, image, segmap, N_AUGMENT)
            for img_aug, sg_aug in zip(image_aug, segmap_aug):
                sg_map = sg_aug.draw(size=image_shape)[0]
                sg_map = ((sg_map[..., 0] != 0)*255).astype(np.uint8)
                images_augmented.append(img_aug)
                masks_augmented.append(sg_map)

        images_augmented = np.array(images_augmented)
        masks_augmented = np.array(masks_augmented)
        np.savez_compressed(os.path.join('Data', 'Image', 'Augmented', d.lower() +
                                         '_augmented'), images=images_augmented, masks=masks_augmented)


# %% MAIN
if __name__ == "__main__":
    main()
