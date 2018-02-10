

'''
icorrectly named unet... it is actually a 64x64 input image since the 50x50 ones did not slice correctly

'''
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras import regularizers
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization

from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
import cv2

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

#TRAIN_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_train/'
#TEST_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_test/'
TRAIN_PATH = 'C:/Users/micha/Desktop/2018_dsb/input/'

model_names = 'deeper_unet_2_3.h5'
save_names = 'deeper_unet_2_3.csv'

sub_name ='C:/Users/micha/Desktop/2018_dsb/submission_files/sub-'+save_names
final_sub_name ='C:/Users/micha/Desktop/2018_dsb/submission_files/final_sub-'+save_names

save_name_file = 'C:/Users/micha/Desktop/2018_dsb/models/model-'+model_names
final_model = 'C:/Users/micha/Desktop/2018_dsb/models/final_model-'+model_names

patience = 4
batch_size_n = 64
epoch_n = 100
val_hold_out = 0.05 #with lrger set might as well keep more samples....?
learning_rate =1e-4
l2_reg = .0001
decay_ = learning_rate/epoch_n


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH+'ta/'))[2]
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
	path = TRAIN_PATH+'ta/'
	img = imread(path +id_)[:,:,:IMG_CHANNELS]
	X_train[n] = img

	mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
	for mask_file in [id_]:
		mask_ = cv2.imread(TRAIN_PATH+'sub/'+ mask_file, 0)
		mask_ = cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
		mask_ = mask_[:, :, np.newaxis]
		mask = np.maximum(mask, mask_)

	Y_train[n] = mask

print('Done!')

# Check if training data looks all right
#ix = random.randint(0, len(train_ids))
#imshow(X_train[ix])
#plt.show()
#imshow(np.squeeze(Y_train[ix]))
#plt.show()

'''
Need to fix this part
'''

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


# Build U-Net model    
#added regularization since conv layers have small numbers of nodes the effect is smaller
def get_unet_256(input_shape=(64, 64, 3),
				 num_classes=1):

	inputs = Input(shape=input_shape)

	'''
	#50
	down0 = Conv2D(32, (3, 3), padding='same')(inputs)
	down0 = BatchNormalization()(down0)
	down0 = Activation('relu')(down0)
	down0 = Conv2D(32, (3, 3), padding='same')(down0)
	down0 = BatchNormalization()(down0)
	down0 = Activation('relu')(down0)
	down0 = Conv2D(32, (3, 3), padding='same')(down0)
	down0 = BatchNormalization()(down0)
	down0 = Activation('relu')(down0)
	down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
	'''
	#25
	down1 = Conv2D(64, (3, 3), padding='same')(inputs)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1 = Conv2D(64, (3, 3), padding='same')(down1)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1 = Conv2D(64, (3, 3), padding='same')(down1)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
	
	#64
	down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2 = Conv2D(128, (3, 3), padding='same')(down2)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2 = Conv2D(128, (3, 3), padding='same')(down2)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
	# 32

	#6
	down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)

	down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
	# 16

	#3
	down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)

	down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
	# 8

	center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	center = Conv2D(1024, (3, 3), padding='same')(center)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	# center

	up4 = UpSampling2D((2, 2))(center)
	up4 = concatenate([down4, up4], axis=3)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)

	# 16

	up3 = UpSampling2D((2, 2))(up4)
	up3 = concatenate([down3, up3], axis=3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)

	# 32

	up2 = UpSampling2D((2, 2))(up3)
	up2 = concatenate([down2, up2], axis=3)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	# 64
	

	up1 = UpSampling2D((2, 2))(up2)
	up1 = concatenate([down1, up1], axis=3)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	'''
	# 128
	up0 = UpSampling2D((2, 2))(up1)
	up0 = concatenate([down0, up0], axis=3)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	'''
	# 256
	# 256

	classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

	model = Model(inputs=inputs, outputs=classify)

	model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=['binary_crossentropy'])

	return model


model = get_unet_256()

#model = load_model(final_model,custom_objects={'bce_dice_loss': bce_dice_loss})

model.summary()

#load old models if restarting runs
# Fit model
#earlystopper = EarlyStopping(patience=patience, verbose=1)
callbacks = [EarlyStopping(monitor='val_loss',
						   patience=5,
						   verbose=1,
						   min_delta=1e-4),
			 ReduceLROnPlateau(monitor='val_loss',
							   factor=0.1,
							   patience=2,
							   verbose=1,
							   epsilon=1e-4),
			 ModelCheckpoint(monitor='val_loss',
							 filepath=save_name_file,
							 save_best_only=True),
			 ModelCheckpoint(final_model),
				 TensorBoard(log_dir='logs')]

#checkpointer = ModelCheckpoint(save_name_file, verbose=1, save_best_only=True)
#ever_epoch_checkpt = ModelCheckpoint(final_model)
results = model.fit(X_train, Y_train, validation_split=val_hold_out, batch_size=batch_size_n, epochs=epoch_n,
					callbacks=callbacks, 
					verbose = 1)#callbacks=[earlystopper, checkpointer])


#########################################################################################################

## save final model
model.save(final_model)
'''
#make predictions
# Predict on train, val and test
model = load_model(save_name_file, custom_objects={'bce_dice_loss': bce_dice_loss})

#model = load_model(save_name_file, custom_objects={'bce_dice_loss': bce_dice_loss})
#preds_train = model.predict(X_train[:int(X_train.shape[0]*0.90)], verbose=1)
#preds_val = model.predict(X_train[int(X_train.shape[0]*0.90):], verbose=1)
preds_test = model.predict(X_test, verbose=1)                                               

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
	preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))


'''
'''
## Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()




# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
'''
'''
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
	dots = np.where(x.T.flatten() == 1)[0]
	run_lengths = []
	prev = -2
	for b in dots:
		if (b>prev+1): run_lengths.extend((b + 1, 0))
		run_lengths[-1] += 1
		prev = b
	return run_lengths

def prob_to_rles(x, cutoff=0.5):
	lab_img = label(x > cutoff)
	for i in range(1, lab_img.max() + 1):
		yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
	rle = list(prob_to_rles(preds_test_upsampled[n]))
	rles.extend(rle)
	new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(sub_name, index=False)


#### Final model
model = load_model(final_model, custom_objects={'bce_dice_loss': bce_dice_loss})

preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
	preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
	rle = list(prob_to_rles(preds_test_upsampled[n]))
	rles.extend(rle)
	new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(final_sub_name, index=False)

'''