import os
import sys
import random
import warnings

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
import matplotlib.pyplot as plt
from skimage import transform
import os
import shutil
from tqdm import tqdm
import cv2


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

TRAIN_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_train/'
TEST_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_test/'
aug_path = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_aug_train2/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def read_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel 
    # by 'William Cukierski'
    image_file = TRAIN_PATH+"{}/images/{}.png".format(image_id,image_id)
    mask_file = TRAIN_PATH+"{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    #masks = skimage.io.imread_collection(mask_file).concatenate()    
    height, width, _ = image.shape
    #num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)

    path = TRAIN_PATH + image_id

    mask = np.zeros((height, width, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (height, width), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

    #for index in range(0, num_masks):
    #    labels[masks[index] > 0] = index + 1
    return image, mask

def data_aug(image,label,angel=30,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label


#image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
print(train_ids[0])
#image_id = TRAIN_PATH+"{}/images/{}.png".format(train_ids[0],train_ids[0])
image, labels = read_image_labels(train_ids[0])
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(np.squeeze(labels))
#plt.show()
#plt.imshow(labels)

new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
plt.subplot(223)
plt.imshow(new_image)
plt.subplot(224)
plt.imshow(np.squeeze(new_labels))

plt.show()

def make_data_augmentation(image_ids,split_num):
    for ax_index, image_id in enumerate(image_ids):
        image,labels = read_image_labels(image_id)
        if not os.path.exists(aug_path+"{}/".format(image_id)):
        	os.makedirs(aug_path+"{}/".format(image_id))
        # also save the original image in augmented file
        if not os.path.exists(aug_path+"{}/images/".format(image_id)):
        	os.makedirs(aug_path+"{}/images/".format(image_id))

        if not os.path.exists(aug_path+"{}/masks/".format(image_id)):
        	os.makedirs(aug_path+"{}/masks/".format(image_id))

        plt.imsave(fname=aug_path+"{}/images/{}.png".format(image_id,image_id), arr = image)
        plt.imsave(fname=aug_path+"{}/masks/{}.png".format(image_id,image_id),arr = np.squeeze(labels))

        for i in range(split_num):
        	#make a directory for each thing
        	if not os.path.exists(aug_path+"{}_{}/".format(image_id,i)):
        		os.makedirs(aug_path+"{}_{}//".format(image_id,i))

        	if not os.path.exists(aug_path+"{}_{}/images/".format(image_id,i)):
        		os.makedirs(aug_path+"{}_{}/images/".format(image_id,i))

        	if not os.path.exists(aug_path+"{}_{}/masks/".format(image_id,i)):
        		os.makedirs(aug_path+"{}_{}/masks/".format(image_id,i))

        	new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)


        	aug_img_dir = aug_path+"{}_{}/images/{}_{}.png".format(image_id,i,image_id,i)
        	aug_mask_dir = aug_path+"{}_{}/masks/{}_{}.png".format(image_id,i,image_id,i)

        	plt.imsave(fname=aug_img_dir, arr = new_image)
        	plt.imsave(fname=aug_mask_dir,arr = np.squeeze(new_labels))

def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        if os.path.exists(TRAIN_PATH+"{}/augs/".format(image_id)):
            shutil.rmtree(TRAIN_PATH+"{}/augs/".format(image_id))
        if os.path.exists(TRAIN_PATH+"{}/augs_masks/".format(image_id)):
            shutil.rmtree(TRAIN_PATH+"{}/augs_masks/".format(image_id))


#image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
split_num = 40


make_data_augmentation(train_ids,split_num)
#clean_data_augmentation(image_ids)