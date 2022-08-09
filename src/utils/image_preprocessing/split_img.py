"""
Implementation for Multi-view Vision Transformer for course IN2392.

Created by:
    Boris Burkalo, TUM/VUT Brno
    Mohammed Said Derbel, TUM
    Alexandre Lutt, TUM/INPC France
    Ludwig Gräf, TUM
"""

import numpy as np
import os
from PIL import Image
import torch

def split_image(img_path, patch_size=4, transform=None):
	im = Image.open(img_path)
	im = np.asarray(transform(im)) if transform else np.asarray(im)
	assert im.shape[0] == im.shape[1]
	
	M = N = im.shape[0] // patch_size

	# Source: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
	tiles = np.array([im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)])

	return tiles
	
def split_imgs(folder_path):
    
    model_patches = []
    for (root, dirs, fns) in os.walk(folder_path):
        for fn in fns:
            if '.png' in fn:
                model_patches.append(split_image(os.path.join(root, fn)))
    
    return np.asarray(model_patches)
    
if __name__ == '__main__':
	## test with squared source img
	img_filename = 'squared_test_img.jpeg'
	split_imgs(img_filename)

	## test with non-squared source img
	img_filename = 'non_squared_test_img.jpeg'
	split_image(img_filename)