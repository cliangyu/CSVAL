import os
import csv
import json
import copy
import keras
import random
import shutil
import tensorflow as tf
from keras import backend as K
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.draw import polygon
import segmentation_models as sm
import cv2
import copy
import matplotlib
from scipy.stats import entropy
from datetime import datetime

def iou(im1, im2):
    overlap = (im1>0.5) * (im2>0.5)
    union = (im1>0.5) + (im2>0.5)
    return overlap.sum()/float(union.sum())
        
def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1>0.5).astype(np.bool)
    im2 = np.asarray(im2>0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

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

# Custom loss function
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def roc_auc_score(gt, pr):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pr, pos_label=1)
    return metrics.auc(fpr, tpr)

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []

    for i in range(classCount):
        outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))

    return outAUROC

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def convert2gray(x):
    x_gray = []
    for i in range(x.shape[0]):
        x_gray.append(rgb2gray(x[i]))
    return np.array(x_gray)

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = convert2gray(x_train)
    x_test = convert2gray(x_test)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    return x_train, y_train, x_test, y_test

def load_mnist():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    return x_train, y_train, x_test, y_test

def preprocess_image(im, gt, input_rows=None, input_cols=None, nb_class=None):
    x = []
    for i in tqdm(range(im.shape[0])):
        img = im[i]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)

        img = resize(img, (input_rows, input_cols, 3))
        x.append(img)
    
    return np.array(x), keras.utils.to_categorical(gt, nb_class)

def preprocess_cifar10_image(im, gt, input_rows=None, input_cols=None, nb_class=None):
    x = []
    for i in tqdm(range(im.shape[0])):
        img = resize(im[i], (input_rows, input_cols, 3))
        x.append(img)
    return np.array(x), keras.utils.to_categorical(gt, nb_class)

def check_unique_value(y, num_unique):
    unique_value = np.unique(y)
    if len(unique_value) == num_unique:
        return True
    else:
        return False

def generate_medmnist_image(x, input_rows=96, input_cols=96,
                            reverse_color=0.1, fade_color=0.1,
                           ):

    if random.random() < 0.5:
        x = np.flip(x, axis=random.choice([0, 1]))
    # if random.random() < 0.2:
    #     x = np.transpose(x, axes=(1,0))
    if random.random() < 0.5:
        x = np.rot90(x, k=random.choice([1,2,3]))

    if random.random() < reverse_color:
        x = 1.0 - x
        
    if random.random() < fade_color:
        x = (random.random() * 0.8 + 0.2) * x

    x = resize(x, (input_rows, input_cols))

    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=-1)
        x = np.concatenate((x, x, x), axis=-1)

    x = resize(x, (input_rows, input_cols, 3))
    
    x[x > 1.] = 1
    x[x < 0.] = 0
    
    return x

def generate_medmnist_pair(x, y, config):
    while True:
        img = np.zeros((config.batch_size, config.input_rows, config.input_cols, config.input_deps), dtype='float')
        gt = np.zeros((config.batch_size, ), dtype='int')
        for i in range(config.batch_size):
            ind = random.randint(0, x.shape[0]-1)
            img[i] = generate_medmnist_image(x[ind], 
                                             input_rows=config.input_rows, 
                                             input_cols=config.input_cols,
                                            )
            gt[i] = y[ind]

        gt = keras.utils.to_categorical(gt, config.nb_class)

        yield (img, gt)

def setup_classification_model(config, suffix=''):
    
    if config.arch == 'Linknet':
        if config.init == 'scratch':
            print('Learning Models from Scratch')
            base_model = sm.Linknet(backbone_name=config.backbone,
                                    encoder_weights=None,
                                    decoder_block_type='upsampling',
                                    classes=1,
                                    activation='sigmoid')
        elif config.init == 'imagenet':
            print('Fine-tuning Models ImageNet')
            base_model = sm.Linknet(backbone_name=config.backbone,
                                    encoder_weights='imagenet',
                                    decoder_block_type='upsampling',
                                    classes=1,
                                    activation='sigmoid')
        
    elif config.arch == 'Unet':
        if config.init == 'scratch':
            print('Learning Models from Scratch')
            base_model = sm.Unet(backbone_name=config.backbone,
                                 encoder_weights=None,
                                 decoder_block_type='upsampling',
                                 classes=1,
                                 activation='sigmoid')
        elif config.init == 'imagenet':
            print('Fine-tuning Models ImageNet')
            base_model = sm.Unet(backbone_name=config.backbone,
                                 encoder_weights='imagenet',
                                 decoder_block_type='upsampling',
                                 classes=1,
                                 activation='sigmoid')
        
    if config.backbone == 'inceptionresnetv2':
        x = base_model.get_layer('conv_7b_ac').output
    x = keras.layers.GlobalAveragePooling2D()(x)
        
    x = keras.layers.Dense(256, activation='relu')(x)
    if config.nb_class > 1:
        output = keras.layers.Dense(config.nb_class, activation='softmax')(x)
    else:
        output = keras.layers.Dense(config.nb_class, activation='sigmoid')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
        
    if config.nb_class > 1:
        loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy'
        
    model.compile(optimizer=keras.optimizers.SGD(lr=config.lr),
                  loss=loss, 
                  metrics=["accuracy", 
                           loss],
                 )
    
    if os.path.exists(os.path.join(config.model_path, config.exp_name+suffix+".txt")):
        os.remove(os.path.join(config.model_path, config.exp_name+suffix+".txt"))
    with open(os.path.join(config.model_path, config.exp_name+suffix+".txt"),'w') as fh:
        model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

    shutil.rmtree(os.path.join(config.logs_path, config.exp_name+suffix), ignore_errors=True)
    if not os.path.exists(os.path.join(config.logs_path, config.exp_name+suffix)):
        os.makedirs(os.path.join(config.logs_path, config.exp_name+suffix))
    tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(config.logs_path, config.exp_name+suffix),
                             histogram_freq=0,
                             write_graph=True, 
                             write_images=True,
                            )
    tbCallBack.set_model(model)    

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_'+loss, 
                                                   patience=config.patience, 
                                                   verbose=config.verbose,
                                                   mode='min',
                                                  )
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(config.model_path, config.exp_name+suffix+".h5"),
                                                  monitor='val_'+loss, 
                                                  verbose=config.verbose, 
                                                  save_best_only=True, 
                                                  mode='min',
                                                 )
    lrate_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_'+loss, factor=0.5, patience=int(0.8*config.patience),
                                        min_delta=0.0001, min_lr=1e-6, verbose=1)
    callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]
    
    return model, callbacks

def setup_classification_model_inference(config):
    
    if config.arch == 'Linknet':
        base_model = sm.Linknet(backbone_name=config.backbone,
                                encoder_weights=None,
                                decoder_block_type='upsampling',
                                classes=1,
                                activation='sigmoid')
        
    elif config.arch == 'Unet':
        base_model = sm.Unet(backbone_name=config.backbone,
                             encoder_weights=None,
                             decoder_block_type='upsampling',
                             classes=1,
                             activation='sigmoid')
        
    if config.backbone == 'inceptionresnetv2':
        x = base_model.get_layer('conv_7b_ac').output
    x = keras.layers.GlobalAveragePooling2D()(x)
        
    x = keras.layers.Dense(256, activation='relu')(x)
    if config.nb_class > 1:
        output = keras.layers.Dense(config.nb_class, activation='softmax')(x)
    else:
        output = keras.layers.Dense(config.nb_class, activation='sigmoid')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
        
    if config.nb_class > 1:
        loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy'
        
    model.compile(optimizer=keras.optimizers.SGD(lr=config.lr),
                  loss=loss, 
                  metrics=["accuracy", 
                           loss],
                 )
    
    return model
