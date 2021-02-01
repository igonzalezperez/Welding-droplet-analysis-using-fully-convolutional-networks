'''
Augment data
'''
import os
import glob
import numpy as np
import cv2
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt
DATASET = ('Globular', 'Spray')
N_AUGMENT = 5

if DATASET == 'Spray':
    IMG_WIDTH = 352
    IMG_HEIGHT = 352
elif DATASET == 'Globular':
    IMG_WIDTH = 352
    IMG_HEIGHT = 296


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


def main():
    '''
    Load original images and masks, augment them and save as images.
    '''
    for d in DATASET:
        data = np.load(os.path.join('HSV Frames', 'npz', 'labelbox',
                                    d.lower(), 'segmented_data.npz'))
        images = data['images']
        masks = data['masks'].astype(bool)
        image_shape = images.shape[1:3]

        # Define our augmentation pipeline.
        seq = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
            iaa.Sharpen((0.0, 1.0)),       # sharpen the image
            # rotate by -45 to 45 degrees (affects segmaps)
            iaa.Affine(rotate=(-45, 45)),
            # apply water effect (affects segmaps)
            iaa.ElasticTransformation(alpha=50, sigma=5)
        ], random_order=True)

        images_augmented, masks_augmented = ([], [])
        for image, mask in zip(images, masks):
            segmap = SegmentationMapsOnImage(mask, shape=image_shape)
            image_aug, segmap_aug = augment_data(seq, image, segmap, N_AUGMENT)
            for img_aug, sg_aug in zip(image_aug, segmap_aug):
                sg_map = sg_aug.draw(size=image_shape)[0]
                sg_map = ((sg_map[..., 0] != 0)*255).astype(np.uint8)
                images_augmented.append(img_aug)
                masks_augmented.append(sg_map)

        images_augmented = np.array(images_augmented)
        masks_augmented = np.array(masks_augmented)
        np.savez(os.path.join('HSV Frames', 'npz', 'augmented', d +
                              '_augmented'), images=images_augmented, masks=masks_augmented)


if __name__ == "__main__":
    main()
