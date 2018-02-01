'''
Use this code to normalize the image files in one directory, 

call it once on the image and the second time on the test set
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imsave

grid_size = 8

save_path = 'E:/2018_dsb/input/stage1_aug_train4/'

TRAIN_PATH = 'E:/2018_dsb/input/2k_aug/'
TEST_PATH = 'E:/2018_dsb/input/stage1_test_normalized/'

main_path = save_path
aug_path = main_path


# Get train and test IDs
main_ids = next(os.walk(main_path))[1]

def rgb_clahe_justl(in_rgb_img): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    return clahe.apply(lab[:,:,0])

#normalize images. unsure if it is better than just making it black and white? 
# but worth testing since it will probably boost final model performance
for ax_index, image_id in enumerate(main_ids,1):
	if ax_index % 1000 == 0:
		print(ax_index)

	#normalize images
	image_file = main_path+"{}/images/{}.png".format(image_id,image_id)
	image_file = imread(image_file)

	image_file = rgb_clahe_justl(image_file)
	normalizer = lambda x: 255-x if x.mean()>127 else x
	image_file = normalizer(image_file)
	imsave(aug_path+"{}/images/{}.png".format(image_id,image_id), image_file)


