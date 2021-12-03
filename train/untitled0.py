# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:23:26 2021

@author: LIU, Mengqi
"""
############################################################
#放到.\train目录下
############################################################
############################################################ 直接运行

import sys
from os.path import join, dirname, realpath
current_dir = dirname(realpath(__file__))
sys.path.insert(0, dirname(current_dir))
import os
import imp
import logging
import random
import timeit
import datetime
import numpy as np
import tensorflow as tf
from utils.logs import set_logging, DebugFolder
from config_path import PROSTATE_LOG_PATH, POSTATE_PARAMS_PATH
from pipeline.train_validate import TrainValidatePipeline
from pipeline.one_split import OneSplitPipeline
from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from pipeline.LeaveOneOut_pipeline import LeaveOneOutPipeline
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import scipy.sparse
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from data.data_access import Data
from model.model_factory import get_model
from pipeline.pipe_utils import get_model_id, get_coef_from_model, get_balanced
from preprocessing import pre
from utils.evaluate import evalualte_survival, evalualte_classification_binary, evalualte_regression
from utils.plots import generate_plots, plot_roc, plot_prc, save_confusion_matrix
# timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())
from utils.rnd import set_random_seeds

from keras import Input
from keras import Model # kears.engine
from keras.layers import Dense, Dropout, Lambda, Concatenate
from keras.regularizers import l2

from data.data_access import Data
from data.pathways.gmt_pathway import get_KEGG_map
from model.builders.builders_utils import get_pnet
from model.layers_custom import f1, Diagonal, SparseTF
from model.model_utils import print_model, get_layers
import keras.backend as K
from sklearn.utils.class_weight import compute_class_weight 
import math


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

random_seed = 234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed) # set_random_seed

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

params_file_list = []

# pnet
params_file_list.append('./pnet/onsplit_average_reg_10_tanh_large_testing')

for params_file in params_file_list:
    log_dir = join(PROSTATE_LOG_PATH, params_file)
    log_dir = log_dir
    set_logging(log_dir)
    params_file = join(POSTATE_PARAMS_PATH, params_file)
    logging.info('random seed %d' % random_seed)
    params_file_full = params_file + '.py'
    # print(params_file_full)
    params = imp.load_source(params_file, params_file_full)
    DebugFolder(log_dir)

task=params.task
data_params=params.data
model_params=params.models
pre_params=params.pre
feature_params=params.features
pipeline_params=params.pipeline,
exp_name=log_dir
eval_dataset = 'test'


test_scores = []
model_names = []
model_list = []
y_pred_test_list = []
y_pred_test_scores_list = []
y_test_list = []
fig = plt.figure()
fig.set_size_inches((10, 6))
#print(self.data_params)
#print('data_params', data_params)
data_params =data_params[0]
data_id = data_params['id']
logging.info('loading data....')
print("+++++++++++++++++++++++++")
print(data_params)
data = Data(**data_params)
            # get data
x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

#logging.info('predicting')
if eval_dataset == 'validation':
    x_t = x_validate_
    y_t = y_validate_
    info_t = info_validate_
else:
    x_t = np.concatenate((x_test_, x_validate_))
    y_t = np.concatenate((y_test_, y_validate_))
    info_t = info_test_.append(info_validate_)

logging.info('x_train {} y_train {} '.format(x_train.shape, y_train.shape))
logging.info('x_test {} y_test {} '.format(x_t.shape, y_t.shape))

def preprocess(x_train, x_test):
    #logging.info('preprocessing....')
    proc = pre.get_processor(pre_params)
    if proc:
        proc.fit(x_train)
        x_train = proc.transform(x_train)
        x_test = proc.transform(x_test)

        if scipy.sparse.issparse(x_train):
            x_train = x_train.todense()
            x_test = x_test.todense()
    return x_train, x_test

x_train, x_test = preprocess(x_train, x_t)


m = model_params[0]
print(m)
model_params_ = deepcopy(m)
set_random_seeds(random_seed=20080808)

p = m['params']['model_params']
#del p['build_fn']

#========================= 构造模型 =====================#
from model.builders.prostate_models import build_pnet2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

model = build_pnet2(**p)
feature_names = model[1]
model = model[0]

pid = os.getpid()
timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}-{0:%S}'.format(datetime.datetime.now())
debug_folder = DebugFolder().get_debug_folder()
save_filename = os.path.join(debug_folder, m['params']['fitting_params']['save_name'] + str(pid) + timeStamp)
reduce_lr = m['params']['fitting_params']['reduce_lr']
monitor = m['params']['fitting_params']['monitor']
select_best_model = m['params']['fitting_params']['select_best_model']
feature_importance = m['params']['feature_importance']
nb_epoch = m['params']['fitting_params']['epoch']

period = 10
save_gradient = m['params']['fitting_params']['save_gradient']
early_stop = m['params']['fitting_params']['early_stop']
lr = m['params']['fitting_params']['lr']
if 'reduce_lr_after_nepochs' in m['params']['fitting_params']:
    reduce_lr_after_nepochs = True
    reduce_lr_drop = m['params']['fitting_params']['reduce_lr_after_nepochs']['drop']
    reduce_lr_epochs_drop = m['params']['fitting_params']['reduce_lr_after_nepochs']['epochs_drop']
else:
    reduce_lr_after_nepochs = False



def get_callbacks(X_train, y_train):
    callbacks = []
    if reduce_lr:
        reduce_lr_ = ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                          patience=2, min_lr=0.000001, verbose=1, mode='auto')
        logging.info("adding a reduce lr on Plateau callback%s " % reduce_lr_)
        callbacks.append(reduce_lr)

    if select_best_model:
        saving_callback = ModelCheckpoint(save_filename, monitor=monitor, verbose=1, save_best_only=True,
                                              mode='max')
        logging.info("adding a saving_callback%s " % saving_callback)
        callbacks.append(saving_callback)

    if save_gradient:
        saving_gradient = GradientCheckpoint(save_filename, feature_importance, X_train, y_train,
                                                 nb_epoch,
                                                 feature_names, period=period)
        logging.info("adding a saving_callback%s " % saving_gradient)
        callbacks.append(saving_gradient)

    if early_stop:
        # early_stop = EarlyStopping(monitor=self.monitor, min_delta=0.01, patience=20, verbose=1, mode='min', baseline=0.6, restore_best_weights=False)
        early_stop_ = FixedEarlyStopping(monitors=[monitor], min_deltas=[0.0], patience=10, verbose=1,
                                            modes=['max'], baselines=[0.0])
        callbacks.append(early_stop_)

    if reduce_lr_after_nepochs:
        # learning rate schedule
        def step_decay(epoch, init_lr, drop, epochs_drop):
            initial_lrate = init_lr
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        from functools import partial
        step_decay_part = partial(step_decay, init_lr=lr, drop=reduce_lr_drop,
                                      epochs_drop=reduce_lr_epochs_drop)
        lr_callback = LearningRateScheduler(step_decay_part, verbose=1)
        callbacks.append(lr_callback)
    return callbacks

callbacks = get_callbacks(x_train, y_train)

class_weight = m['params']['fitting_params']['class_weight']
if class_weight == 'auto':
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes = classes, y=y_train.ravel())
    class_weights = dict(zip(classes, class_weights))
else:
    class_weights = class_weight

# speical case of survival
if y_train.dtype.fields is not None:
    y_train = y_train['time']


x_val = x_validate_
y_val = y_validate_
debug = m['params']['fitting_params']['debug']

#if debug:
    # train on 80 and validate on 20, report validation and training performance over epochs
 #   logging.info('dividing training data into train and validation with split 80 to 20')
 #   training_data, validation_data = get_validation_set(x_train, y_train, test_size=0.2)
 #   X_train, y_train = training_data
 #   x_val, y_val = validation_data

if 'n_outputs' in m['params']['fitting_params']:
    n_outputs = m['params']['fitting_params']['n_outputs']
else:
    n_outputs = 1

if n_outputs > 1:
    y_train = [y_train] * n_outputs
    y_val = [y_val] * n_outputs

if not x_val is None:
    validation_data = [x_val, y_val]
else:
    validation_data = []

x_val = validation_data[0]
y_val = validation_data[1]
batch_size = m['params']['fitting_params']['batch_size']
verbose = m['params']['fitting_params']['verbose']
shuffle = m['params']['fitting_params']['shuffle']


###########################################################################

class_weights = {0: [0.7458410351201479]*6, 1: [1.5169172932330828]*6}

print(x_train.shape)
##[<tf.Tensor 'mul_2:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'mul_5:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'mul_8:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'mul_11:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'mul_14:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'mul_17:0' shape=(6,) dtype=float32>, 
# <tf.Tensor 'AddN:0' shape=() dtype=float32>]
##
class_weights = {0: 0.7458410351201479, 1: 1.5169172932330828} 
sample_weight = [class_weights[1] if i==1 else class_weights[0] for i in y_train[0]]

history = model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=verbose, callbacks=callbacks,
                    shuffle=shuffle, sample_weight = np.array(sample_weight))
model(x_train, y_train) #test











