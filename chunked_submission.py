
import os
import sys
import random
import warnings
import shutil
import cv2

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import img_as_uint


from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.losses import binary_crossentropy

import tensorflow as tf
from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

def slice_image_cubes(img_path,save_path,im_size):
	from PIL import Image
	img = Image.open(img_path)
	(imageWidth, imageHeight)=img.size

	img = img.resize((round(imageWidth/im_size)*im_size,round(imageHeight/im_size)*im_size))
	(imageWidth, imageHeight)=img.size

	from PIL import Image
	gridx=im_size
	gridy=im_size
	rangex=imageWidth/gridx
	rangey=imageHeight/gridy

	print(rangex*rangey)
	for y in range(int(rangey)):
		for x in range(int(rangex)):
			bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
			slice_bit=img.crop(bbox)
			slice_bit.save(save_path+str(y).zfill(2)+'_'+str(x).zfill(2)+'.png', optimize=True, bits=6)


def merge_right(image1, image2):
	"""Merge two images into one, displayed side by side
	:param file1: path to first image file
	:param file2: path to second image file
	:return: the merged Image object
	"""


	(width1, height1) = image1.size
	(width2, height2) = image2.size

	result_width = width1 + width2
	result_height = max(height1, height2)

	result = Image.new('RGB', (result_width, result_height))
	result.paste(im=image1, box=(0, 0))
	result.paste(im=image2, box=(width1, 0))
	return result

#os.chdir("C:/Users/585000/Desktop/stitch test/")


#relies on format ##_##.png
def recombine_cropped_images(image_name,image_dir,save_dir):
	file_list = []
	#for file in glob.glob(image_dir+"*.png"):
	#    file_list.append(file)
	#print(file_list)
	#print(image_name,image_dir,save_dir)
	#print('hello world')
	file_list = next(os.walk(image_dir))[2]
	#print(file_list)
	row_max_list = []
	for file in file_list:
		if int(file[:2]) not in row_max_list:
			row_max_list.append(int(file[:2]))
	#max(row_max_list)
	#min(row_max_list)
	width_max_list = []
	for file in file_list:
		if int(file[-6:-4]) not in width_max_list:
			width_max_list.append(int(file[-6:-4]))

	for row in range(min(row_max_list),max(row_max_list)+1):

		for column in range(min(width_max_list), max(width_max_list)+1):
			if int(column) == 0:
				image1 = Image.open(image_dir+'{}_00.png'.format(str(row).zfill(2)))
				#print('C:/Users/585000/Desktop/stitch test/{}_00.png'.format(str(row).zfill(2)))

			else:
				image2 = Image.open(image_dir+str(row).zfill(2)+'_{}.png'.format(str(column).zfill(2)))
				image1 = merge_right(image1,image2)

			#image1.save('testmerge{}.png'.format(str(row).zfill(2)))
		if int(row) == 0:
			rebuilding_row = image1

		else:
			rebuilding_row = np.vstack([rebuilding_row,image1])


	rebuilding_row = Image.fromarray( rebuilding_row)
	rebuilding_row.save(save_dir+image_name+'.png')



# Define IoU metric
def mean_iou(y_true, y_pred):
	prec = []
	for t in np.arange(0.5, 1.0, 0.05):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return K.mean(K.stack(prec), axis=0)

def rle_encoding(x):
	dots = np.where(x.T.flatten() == 1)[0]
	run_lengths = []
	prev = -2
	for b in dots:
		if (b>prev+1): run_lengths.extend((b + 1, 0))
		run_lengths[-1] += 1
		prev = b
	return run_lengths

def prob_to_rles(x, cutoff=0.50):
	lab_img = label(x > cutoff)
	for i in range(1, lab_img.max() + 1):
		yield rle_encoding(lab_img == i)
#def slice_image_cubes(img_path,save_path,im_size):

#slice_image_cubes('C:/Users/585000/Desktop/RWBY_Cover.jpg','C:/Users/585000/Desktop/cubed_test_1/',20)


#def recombine_cropped_images(image_name,image_dir,save_dir):
#recombine_cropped_images('Rwby_Cover_merged','C:/Users/585000/Desktop/cubed_test_1/','C:/Users/585000/Desktop/')

######## MODEL TESTING
if __name__ == "__main__":

	# Set some parameters
	IMG_WIDTH = 128
	IMG_HEIGHT = 128
	IMG_CHANNELS = 3

	#TRAIN_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_train/'
	TEST_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_test_normalized/'

	test_holder = 'C:/Users/micha/Desktop/2018_dsb/input/test_holder/'
	test_saver = 'C:/Users/micha/Desktop/2018_dsb/input/test_saver/'
	test_pred_holder = 'C:/Users/micha/Desktop/2018_dsb/input/test_pred_holder/'

	del_test_pred_holder = 'C:/Users/micha/Desktop/2018_dsb/input/test_pred_holder'
	del_test_holder ='C:/Users/micha/Desktop/2018_dsb/input/test_holder'

	model_names = '128x128_unet_normalized.h5'
	save_names = '128x128_unet_normalized.csv'

	sub_name ='C:/Users/micha/Desktop/2018_dsb/submission_files/sub-'+save_names
	final_sub_name ='C:/Users/micha/Desktop/2018_dsb/submission_files/final_sub-'+save_names

	best_model = 'C:/Users/micha/Desktop/2018_dsb/models/model-'+model_names
	final_model = 'C:/Users/micha/Desktop/2018_dsb/models/final_model-'+model_names

	patience = 3
	batch_size_n = 1
	epoch_n = 100
	val_hold_out = 0.03 #with larger set might as well keep more samples....?
	learning_rate =0.0001

	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	seed = 42
	random.seed = seed
	np.random.seed = seed

	#read in images
	# make them binary masks
	# process them like standard imag

	
	code to generate compiled masks
	test_ids = next(os.walk(TEST_PATH))[1]

	model = load_model(final_model, custom_objects={'bce_dice_loss': bce_dice_loss})
	
	for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
		path = TEST_PATH + id_
		##def slice_image_cubes(img_path,save_path,im_size):
		slice_image_cubes(path + '/images/' + id_ + '.png',test_holder,IMG_HEIGHT) #slices the test image
		file_list = next(os.walk(test_holder))[2] #get all the images in the test holder file then iterate on them

		for file in file_list:
			X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
			img = imread(test_holder+file)[:,:,:IMG_CHANNELS]
			X_test[0] = img
			

			preds_test = model.predict(X_test, verbose=1,batch_size=1)
			#imshow(X_test[0])
			#plt.show()
			#imshow(np.squeeze(preds_test[0]))
			#plt.show()
			plt.imsave(fname=test_pred_holder+file,arr = np.squeeze(preds_test[0])) #save predictions

		#stitch predictions together

		#def recombine_cropped_images(image_name,image_dir,save_dir):
		#print('why...',test_pred_holder)

		recombine_cropped_images(id_,test_pred_holder,test_saver)


		dirPath = del_test_pred_holder
		fileList = os.listdir(dirPath)
		for fileName in fileList:
 				os.remove(dirPath+"/"+fileName)


		dirPath = del_test_holder
		fileList = os.listdir(dirPath)
		for fileName in fileList:
 				os.remove(dirPath+"/"+fileName)
 			
	es where values are 1 or 0

	mask_ids = next(os.walk(test_saver))[2]
	preds_test_upsampled = []

	for i, id_ in enumerate(mask_ids):
			#path = train_path + id_
			#print(path)
		path = test_saver

		im_gray = cv2.imread(path +id_, 0)
		#print(im_gray.shape)
		thresh = 127
		im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
		
		#cv2.imshow('test',im_bw)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#get the sizes of originak images
		path = TEST_PATH + id_[:-4]
		img = imread(path + '/images/' + id_ )

		preds_test_upsampled.append(resize(np.squeeze(im_gray),
										   (img.shape[0], img.shape[1]), 
										   mode='constant', preserve_range=True))
		

	def rle_encoding(x):
		dots = np.where(x.T.flatten() == 1)[0]
		run_lengths = []
		prev = -2
		for b in dots:
			if (b>prev+1): run_lengths.extend((b + 1, 0))
			run_lengths[-1] += 1
			prev = b
		return run_lengths

	#scale 0-255
	def prob_to_rles(x, cutoff=127):
		lab_img = label(x > cutoff)
		for i in range(1, lab_img.max() + 1):
			yield rle_encoding(lab_img == i)

	new_test_ids = []
	rles = []
	for n, id_ in enumerate(mask_ids):
		idd = id_[:-4]
		rle = list(prob_to_rles(preds_test_upsampled[n]))
		rles.extend(rle)
		new_test_ids.extend([idd] * len(rle))

	print(len(mask_ids))
	print(len(preds_test_upsampled))
	print(len(new_test_ids))

	# Create submission DataFrame
	sub = pd.DataFrame()
	sub['ImageId'] = new_test_ids
	sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	sub.to_csv(sub_name, index=False)


