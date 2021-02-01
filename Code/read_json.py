'''
Read and save labelbox annotated data.
'''
import os
from PIL import Image
import pandas as pd
from labelbox import Client
from imageio import imread
import urllib
import io
import numpy as np
from progressbar import progressbar as progress
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv(os.path.join('Code', 'Scripts', '.env'))
# CLIENT = Client(api_key=API_KEY)
# client = Client(api_key)

DATASET = ('Globular', 'Spray')


def load_and_save_json():
    for d in DATASET:
        json_file = pd.read_json(os.path.join(
            'HSV Frames', 'npz', 'labelbox', d.lower() + '_masks.json'))
        _ids = []
        masks = []
        for i in progress(range(len(json_file))):
            _ids.append(int(json_file['External ID'][i][:-4]))
            annot = json_file['Label'][i]['objects'][0]['instanceURI']

            with urllib.request.urlopen(annot) as url:
                f = io.BytesIO(url.read())
            image = Image.open(f).convert('L')
            masks.append(np.asarray(image))
        images = np.load(os.path.join('HSV Frames', 'npz', d +
                                      '_gray.npz'))['images'][_ids, ...]
        np.savez(os.path.join('HSV Frames', 'npz', 'labelbox', d.lower(),
                              'segmented_data'), images=images, masks=masks)


if __name__ == '__main__':
    load_dotenv()
