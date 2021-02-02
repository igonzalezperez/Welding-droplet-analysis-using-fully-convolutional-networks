'''
Miscellaneous utility functions.
'''
# %% IMPORTS
import math
from typing import List, Generator
from PIL import Image

# %%FUNCTIONS


def chunks(input_list: List, batch_size: int) -> Generator[int, None, None]:
    '''
    Partitions list in chunks of size batch_size. [1,2,3,4,5] -> [[1,2],[3,4],[5]] That is a list
    partitioned with batch_size = 2.
    '''
    batch_size = max(1, batch_size)
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i+batch_size]


def get_concat_h(im1, im2):
    '''
    Horizontaly join two images of same height.
    '''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def upper_round(div: int, num: int) -> int:
    '''
    Returns the difference between num and the closest multiple of div that is larger than num.
    upper_round(3, 10) -> 2
    upper round(4, 27) -> 1
    upper round(5, 25) -> 0
    '''
    x = div*math.ceil(num/div)
    p = x - num
    return p


# %%MAIN
if __name__ == '__main__':
    pass
