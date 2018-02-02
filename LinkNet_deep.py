from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add
from keras.layers.core import Flatten, Reshape

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.morphology import label

import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def make_df(train_path, test_path, img_size):
    print('lets do this!')
    train_ids = next(os.walk(train_path))[1]
    test_ids = next(os.walk(test_path))[1]
    X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    for i, id_ in enumerate(train_ids):
        path = train_path + id_
        #print(path)
        img = cv2.imread(path + '/images/' + id_ + '.png')
        img = cv2.resize(img, (img_size, img_size))
        X_train[i] = img
        mask = np.zeros((img_size, img_size, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (img_size, img_size))
            mask_ = mask_[:, :, np.newaxis]
            mask = np.maximum(mask, mask_)
        Y_train[i] = mask
    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
    sizes_test = []
    for i, id_ in enumerate(test_ids):
        path = test_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_size, img_size))
        X_test[i] = img

    return X_train, Y_train, X_test, sizes_test


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.00001))(input)

    return add([shortcut, residual])

def encoder_block(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_1 = _shortcut(input_tensor, x)

    x = BatchNormalization()(added_1)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    added_2 = _shortcut(added_1, x)

    return added_2

def decoder_block(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1))(x)

    return x

def LinkNet(input_shape=(256, 256, 3), classes=1):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block(input_tensor=x, m=128, n=128)

    encoder_2 = encoder_block(input_tensor=encoder_1, m=128, n=256)

    encoder_3 = encoder_block(input_tensor=encoder_2, m=256, n=512)

    encoder_4 = encoder_block(input_tensor=encoder_3, m=512, n=1024)

    decoder_4 = decoder_block(input_tensor=encoder_4, m=1024, n=512)

    decoder_3_in = add([decoder_4, encoder_3])
    decoder_3_in = Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block(input_tensor=decoder_3_in, m=512, n=256)

    decoder_2_in = add([decoder_3, encoder_2])
    decoder_2_in = Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=256, n=128)

    decoder_1_in = add([decoder_2, encoder_1])
    decoder_1_in = Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=128, n=128)

    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def generator(xtr, xval, ytr, yval, batch_size):
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range = 0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr, seed=7)
    mask_datagen.fit(ytr, seed=7)
    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval, seed=7)
    mask_datagen_val.fit(yval, seed=7)
    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
    


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





if __name__ == "__main__":

    img_size = 256
    batch_size_n = 16
    epoch_n = 200
    val_hold_out = 0.05 #with larger set might as well keep more samples....?
    learning_rate = 1e-4
    decay_ = learning_rate /epoch_n
    momentum = .8

    #TRAIN_PATH = 'E:/2018_dsb/input/stage1_train/'
    TRAIN_PATH = 'E:/2018_dsb/input/stage1_aug_train/'
    TEST_PATH = 'E:/2018_dsb/input/stage1_test/'

    model_names = 'normalized_LinkNet_1070_5_unet.h5'
    save_names = 'normalized_LinkNet_1070_5_unet.csv'

    sub_name ='E:/2018_dsb/submission_files/best_sub-'+save_names
    final_sub_name ='E:/2018_dsb/submission_files/final_sub-'+save_names

    save_name_file = 'E:/2018_dsb/models/best_model-'+model_names
    final_model = 'E:/2018_dsb/models/final_model-'+model_names

    print('building train and val sets')
    X_train, Y_train, X_test, sizes_test = make_df(TRAIN_PATH, TEST_PATH, img_size)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=val_hold_out, random_state=7)
    train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size_n)
    
    print('putting the model together')

    model = LinkNet(input_shape=(256, 256, 3), classes=1)

    #opt = Adam(lr=learning_rate)
    opt = RMSprop(lr=learning_rate)
    #model = load_model('E:/2018_dsb/models/model-1_24_dsbowl2018-11_512_unet.h5',custom_objects={'mean_iou': mean_iou})
    #model = load_model(final_model,custom_objects={'bce_dice_loss': bce_dice_loss})

    #opt = SGD(lr=learning_rate,momentum =momentum,decay=decay_)

    #load old models if restarting runs
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=['binary_crossentropy'])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou]) 

    callbacks = [EarlyStopping(monitor='val_loss',
                           patience=15,
                           verbose=1,
                           min_delta=1e-5),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-5),
             ModelCheckpoint(monitor='val_loss',
                             filepath=save_name_file,
                             save_best_only=True,verbose=1),
             ModelCheckpoint(final_model),
             TensorBoard(log_dir='logs')]


    #checkpointer = ModelCheckpoint(save_name_file, verbose=1, save_best_only=True)
    #every_epoch = ModelCheckpoint(final_model)
    print("show me what you're made of")

    model.fit_generator(train_generator, steps_per_epoch=len(xtr)/batch_size_n, epochs=epoch_n,
                        validation_data=val_generator, validation_steps=len(xval)/batch_size_n,callbacks=callbacks)


    #################################### Evaluation section ####################################


    model.save(final_model)

    model = load_model(save_name_file)

    preds_test = model.predict(X_test, verbose=1)

    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(cv2.resize(preds_test[i], 
                                           (sizes_test[i][1], sizes_test[i][0])))
        
    test_ids = next(os.walk(test_path))[1]
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    
    sub.to_csv(sub_name, index=False)


    model = load_model(final_model)

    preds_test = model.predict(X_test, verbose=1)

    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(cv2.resize(preds_test[i], 
                                           (sizes_test[i][1], sizes_test[i][0])))
        
    test_ids = next(os.walk(test_path))[1]
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    
    sub.to_csv(final_sub_name, index=False)



