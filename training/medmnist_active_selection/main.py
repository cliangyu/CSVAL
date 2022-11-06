#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2022, JOHNS HOPKINS UNIVERSITY. All rights reserved.
# Author: Zongwei Zhou
# Email: zzhou82@jh.edu
# Last modification: May 29, 2022

'''
for run in {2..10}; do for task in mnist10; do for partial in $(seq 0.00045 0.00005 0.001); do sbatch --error=logs/$task-$run-p$partial.out --output=logs/$task-$run-p$partial.out hg.sh $run $task $partial; done; done; done

python -W ignore main.py --gpu 3 --run 1 --task mnist10 --partial 0.0005
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')
import medmnist
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
import keras
print('keras = {}'.format(keras.__version__))
import tensorflow as tf
print('tensorflow-gpu = {}'.format(tf.__version__))
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import random
import copy
import shutil
import math
import argparse
import dataset_without_pytorch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.transform import resize
from glob import glob
from tqdm import tqdm
from sklearn import metrics
from utils import *
from config import *
from medmnist import INFO, Evaluator

parser = argparse.ArgumentParser(description='main')

parser.add_argument('--act', dest='act', default=None, type=str, help="active querying strategy")
parser.add_argument('--arch', dest='arch', default='Linknet', type=str, help="Linknet | Unet")
parser.add_argument('--backbone', dest='backbone', default='inceptionresnetv2', type=str, help="backbone")
parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help="batch size")
parser.add_argument('--gpu', dest='gpu', default=None, type=str, help="gpu index")
parser.add_argument('--init', dest='init', default='scratch', type=str, help="scratch | imagenet")
parser.add_argument('--input_rows', dest='input_rows', default=48, type=int, help="input rows")
parser.add_argument('--input_cols', dest='input_cols', default=48, type=int, help="input cols")
parser.add_argument('--input_deps', dest='input_deps', default=3, type=int, help="input deps")
parser.add_argument('--lr', dest='lr', default=0.1, type=float, help="learning rate")
parser.add_argument('--partial', dest='partial', default=1.0, type=float, help="partial data %")
parser.add_argument('--patience', dest='patience', default=5, type=int, help="patience")
parser.add_argument('--run', dest='run', default=1, type=int, help="multiple trials")
parser.add_argument('--task', dest='task', default='mnistanatomy', type=str, help="mnistanatomy | mnist10")

args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
assert args.task in ['mnistanatomy', 
                     'mnist10', 
                     'bloodmnist',
                     'cifar10',
                     'cifar10lt',
                     'dermamnist', 
                     'octmnist',
                     'pathmnist', 
                     'retinamnist',
                     'chestmnist', 
                     'pneumoniamnist',
                     'breastmnist', 
                     'tissuemnist', 
                     'organamnist',
                     'organcmnist', 
                     'organsmnist',
                    ]
assert args.init in ['scratch', 'imagenet']
assert args.act in ['uncertainty', 
                    'vaal', 
                    'consistency', 
                    'bald', 
                    'coreset', 
                    'margin',
                    'easy',
                    'hard',
                    'ambiguous',
                    'gt_easy',
                    'gt_hard',
                    'gt_ambiguous',
                   ]

info = INFO[args.task]
task = info['task']

config = cold_start_config(args)
config.n_channels = info['n_channels']
config.nb_class = len(info['label'])
config.display()

DataClass = getattr(dataset_without_pytorch, info['python_class'])
train_dataset = DataClass(split='train', download=True)
print(train_dataset)
val_dataset = DataClass(split='val', download=True)
print(val_dataset)
test_dataset = DataClass(split='test', download=True)
print(test_dataset)
x, y = train_dataset[0]

if len(np.array(x).shape) == 2:
    x_train, y_train = np.zeros((len(train_dataset), np.array(x).shape[0], np.array(x).shape[1]), dtype='float'), np.zeros((len(train_dataset), 1), dtype='int')
    x_val, y_val = np.zeros((len(val_dataset), np.array(x).shape[0], np.array(x).shape[1]), dtype='float'), np.zeros((len(val_dataset), 1), dtype='int')
    x_test, y_test = np.zeros((len(test_dataset), np.array(x).shape[0], np.array(x).shape[1]), dtype='float'), np.zeros((len(test_dataset), 1), dtype='int')
elif len(np.array(x).shape) == 3:
    x_train, y_train = np.zeros((len(train_dataset), np.array(x).shape[0], np.array(x).shape[1], np.array(x).shape[2]), dtype='float'), np.zeros((len(train_dataset), 1), dtype='int')
    x_val, y_val = np.zeros((len(val_dataset), np.array(x).shape[0], np.array(x).shape[1], np.array(x).shape[2]), dtype='float'), np.zeros((len(val_dataset), 1), dtype='int')
    x_test, y_test = np.zeros((len(test_dataset), np.array(x).shape[0], np.array(x).shape[1], np.array(x).shape[2]), dtype='float'), np.zeros((len(test_dataset), 1), dtype='int')
else:
    raise

for i in range(len(train_dataset)):
    x, y = train_dataset[i]
    x_train[i] = np.array(x)
    y_train[i] = np.array(y)
x_train = x_train.astype("float32") / 255.0
print("x_train shape: {} | {} ~ {}".format(x_train.shape, np.min(x_train), np.max(x_train)))

for i in range(len(val_dataset)):
    x, y = val_dataset[i]
    x_val[i] = np.array(x)
    y_val[i] = np.array(y)
x_val = x_val.astype("float32") / 255.0
print("x_val shape: {} | {} ~ {}".format(x_val.shape, np.min(x_val), np.max(x_val)))

for i in range(len(test_dataset)):
    x, y = test_dataset[i]
    x_test[i] = np.array(x)
    y_test[i] = np.array(y)
x_test = x_test.astype("float32") / 255.0
print("x_test shape: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))

print(x_train.shape[0], "train samples")
print(x_val.shape[0], "val samples")
print(x_test.shape[0], "test samples")

num_train = int(x_train.shape[0])
num_partial = int(num_train * args.partial)
assert num_partial >= config.nb_class
sample_ind = np.load(os.path.join(config.idx_path, args.task, args.act+'_idx.npy'))
sample_ind = sample_ind[:num_partial]
if check_unique_value(y_train[sample_ind], num_unique=config.nb_class) == False:
    print('[ERROR!] Cannot cover all classes.')
    raise
print(sample_ind)
print(y_train[sample_ind])
print('{:.4f}% of training set: {} images'.format(100*args.partial, num_partial))

suffix = '-'+args.task+'-p'+str(args.partial)+'-a-'+str(args.act)

model, callbacks = setup_classification_model(config, suffix=suffix)
            
while config.batch_size > 1:
    try:
        history = model.fit_generator(generate_medmnist_pair(x_train[sample_ind],
                                                             y_train[sample_ind],
                                                             config=config,
                                                            ),
                                      validation_data=generate_medmnist_pair(x_val,
                                                                             y_val,
                                                                             config=config,
                                                                            ),
                                      validation_steps=max(int(1.0*x_val.shape[0])//config.batch_size, 5),
                                      steps_per_epoch=max(int(1.0*num_partial)//config.batch_size, 50),
                                      epochs=config.nb_epoch,
                                      max_queue_size=config.max_queue_size,
                                      workers=config.workers,
                                      use_multiprocessing=config.use_multiprocessing,
                                      shuffle=True,
                                      verbose=config.verbose,
                                      callbacks=callbacks,
                                     )  
        break
    except tf.errors.ResourceExhaustedError as e:
        config.batch_size = int(config.batch_size - 2)
        print('\n> Batch size = {}'.format(config.batch_size))
            


x_test, y_test = preprocess_image(x_test, y_test, 
                                  input_rows=config.input_rows, 
                                  input_cols=config.input_cols,
                                  nb_class=config.nb_class,
                                 )
model = setup_classification_model_inference(config)
print('Load model from {}'.format(os.path.join(config.model_path, config.exp_name+suffix+".h5")))
model.load_weights(os.path.join(config.model_path, config.exp_name+suffix+".h5"))
p_test = model.predict(x_test, verbose=1)
print('x_test: {} | {} ~ {}'.format(x_test.shape, np.min(x_test), np.max(x_test)))
print('y_test: {} | {} ~ {}'.format(y_test.shape, np.min(y_test), np.max(y_test)))
print('p_test: {} | {} ~ {}'.format(p_test.shape, np.min(p_test), np.max(p_test)))

np.save(os.path.join(config.predict_path, config.exp_name+suffix+".npy"), 
        np.array(p_test, dtype='float16'),
       )
os.remove(os.path.join(config.model_path, config.exp_name+suffix+".h5"))

auc = computeAUROC(y_test, p_test, config.nb_class)
for i in range(config.nb_class):
    print('NUM {}   | AUC = {:.4f}'.format(i, auc[i]))
print('AVERAGE | AUC = {:.4f}'.format(sum(auc) / len(auc)))
