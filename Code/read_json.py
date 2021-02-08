'''
Read and save labelbox annotated data.
'''
# %% IMPORTS
import os
import urllib
import io
import numpy as np
import pandas as pd
from PIL import Image
from progressbar import progressbar as progress
from dotenv import load_dotenv
# from labelbox import Client

# %% VARIABLES
# load api key to read images' urls from json file
load_dotenv(os.path.join('Code', '.env'))
# CLIENT = Client(api_key=API_KEY)
# client = Client(api_key)

DATASET = ('Globular', 'Spray')

# %% FUNCTIONS


def load_and_save_json():
    '''
    Loads exported json file from LabelBox containing urls to each image and segmentation map.
    '''
    for dataset in DATASET:
        json_file = pd.read_json(os.path.join(
            'Data', 'json', dataset.lower() + '_masks.json'))
        _ids = []
        masks = []
        for i in progress(range(len(json_file))):
            _ids.append(int(json_file['External ID'][i][:-4]))
            annot = json_file['Label'][i]['objects'][0]['instanceURI']

            with urllib.request.urlopen(annot) as url:
                image_bytes_url = io.BytesIO(url.read())
            image = Image.open(image_bytes_url).convert('L')
            masks.append(np.asarray(image))
        images = np.load(os.path.join('Data', 'Image', 'Input', dataset.lower() +
                                      '_gray.npz'))['images'][_ids, ...]
        np.savez_compressed(os.path.join('Data', 'Image', 'Labelbox', dataset.lower() +
                                         '_segmented'), images=images, masks=masks)


# %% MAIN
if __name__ == '__main__':
    load_and_save_json()
