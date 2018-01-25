
##### WORK IN PROGRESS 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image

grid_size = 8

save_path = 'E:/2018_dsb/input/stage1_train/normalized_test/'

TRAIN_PATH = 'E:/2018_dsb/input/stage1_train/'
TEST_PATH = 'E:/2018_dsb/input/stage1_test/'


main_path = TRAIN_PATH
aug_path = save_path

# Get train and test IDs
main_ids = next(os.walk(main_path))[1]

def rgb_clahe_justl(in_rgb_img): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    return clahe.apply(lab[:,:,0])


#for train 
for ax_index, image_id in enumerate(main_ids,1):
	if ax_index % 1000 == 0:
		print(ax_index)

	#normalize images
	image_file = main_path+"{}/images/{}.png".format(image_id,image_id)
	image_file = imread(image_file)
	height, width, _ = image_file.shape

	image_file = rgb_clahe_justl(image_file)

	image_file = image_file.point(lambda x: 255-x if x.mean()>127 else x) 

	path = main_path + image_id

	mask = np.zeros((height, width, 1), dtype=np.bool)

	for mask_file in next(os.walk(path + '/masks/'))[2]:

		mask_ = imread(path + '/masks/' + mask_file)

		mask_ = np.expand_dims(resize(mask_, (height, width), mode='constant', 
                                      preserve_range=True), axis=-1)

		mask = np.maximum(mask, mask_)

	#main file
	if not os.path.exists(aug_path+"{}/".format(image_id)):
		os.makedirs(aug_path+"{}/".format(image_id))
		# also save the original image in augmented file
	#image file
	if not os.path.exists(aug_path+"{}/images/".format(image_id)):
		os.makedirs(aug_path+"{}/images/".format(image_id))
	#mask file
	if not os.path.exists(aug_path+"{}/masks/".format(image_id)):
		os.makedirs(aug_path+"{}/masks/".format(image_id))

	#save
	plt.imsave(fname=aug_path+"{}/images/{}.png".format(image_id,image_id), arr = image_file)

	plt.imsave(fname=aug_path+"{}/masks/{}.png".format(image_id,image_id),arr = np.squeeze(mask))


