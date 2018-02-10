
from PIL import Image
import os
import cv2
from skimage.io import imread

grid_size = 8
aug_path = 'C:/Users/micha/Desktop/2018_dsb/input/sub'
# Get train and test IDs
aug_ids = next(os.walk(aug_path))[2]

#print(aug_ids)
def rgb_clahe_justl(in_rgb_img): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    return clahe.apply(lab[:,:,0])

for ax_index, image_id in enumerate(aug_ids,1):
	if ax_index % 500 == 0:
		print(ax_index)
	mask_file = aug_path+"/{}".format(image_id,image_id)
	col = Image.open(mask_file)
	grey = col.convert('L')
	bw = grey.point(lambda x: 0 if x<128 else 255,'1')
	bw.save(mask_file)
	'''
	image_file_path = aug_path+"{}/images/{}.png".format(image_id,image_id)
	image_file = imread(image_file_path)

	image_file = rgb_clahe_justl(image_file)
	normalize = lambda x: 255-x if x.mean()>127 else x
	image_file = flatten(normalize(image_file))
	image_file.save(image_file_path)
	'''

