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

path_1 = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_train-normalized/'
save_path = 'C:/Users/micha/Desktop/2018_dsb/input/chunked_train_128_normalized/'

TRAIN_PATH = 'E:/2018_dsb/input/2k_aug/'
TEST_PATH = 'E:/2018_dsb/input/stage1_test_normalized/'

main_path = path_1
aug_path = main_path


# Get train and test IDs
main_ids = next(os.walk(main_path))[1]


gridx=128
gridy=128

from PIL import Image

for ax_index, image_id in enumerate(main_ids,1):
	if ax_index % 100 == 0:
		print(ax_index)

	#normalize images
	image_file = main_path+"{}/images/{}.png".format(image_id,image_id)
	image_file = Image.open(image_file)
	(imageWidth, imageHeight)=image_file.size

	image_file = image_file.resize((round((imageWidth/gridx))*gridx,round(imageHeight/gridy)*gridy))

	(imageWidth, imageHeight)=image_file.size

	rangex=imageWidth/gridx
	rangey=imageHeight/gridy

	for y in range(int(rangey)):
		for x in range(int(rangex)):
			bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
			slice_bit=image_file.crop(bbox)
			slice_bit.save(save_path+ image_id+str(y).zfill(2)+'_'+str(x).zfill(2)+'.png', optimize=True, bits=6)


	path = path_1 + image_id

	mask = np.zeros((imageHeight, imageWidth, 1), dtype=np.bool)

	for mask_file in next(os.walk(path + '/masks/'))[2]:

		mask_ = imread(path + '/masks/' + mask_file)
		mask_ = np.expand_dims(resize(mask_, (imageHeight, imageWidth), mode='constant', 
									  preserve_range=True), axis=-1)
		mask = np.maximum(mask, mask_)

	plt.imsave(fname='C:/Users/micha/Desktop/2018_dsb/input/normalized_masks_for_chunking/'+"{}.png".format(image_id),arr = np.squeeze(mask))

	mask_file = 'C:/Users/micha/Desktop/2018_dsb/input/normalized_masks_for_chunking/'+"{}.png".format(image_id)
	mask_file = Image.open(mask_file)
	
	for y in range(int(rangey)):
		for x in range(int(rangex)):
			bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
			slice_bit=mask_file.crop(bbox)
			slice_bit.save('C:/Users/micha/Desktop/2018_dsb/input/chunked_masks_128_normalized/'+ image_id+str(y).zfill(2)+'_'+str(x).zfill(2)+'.png', optimize=True, bits=6)







