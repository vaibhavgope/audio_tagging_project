import streamlit as st
import gc
import os
import sys
import time
import pickle
import random
import logging
import datetime as dt
import math
import io

import librosa
import numpy as np
import pandas as pd
import subprocess as sp

from PIL import Image
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import sklearn.metrics

from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import regularizers
from imgaug import augmenters as iaa
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.constraints import max_norm, MinMaxNorm

SEED = 42
BATCH_SIZE = 32
SIZE = 128
USE_MIXUP = True
MIXUP_PROB = 0.275

LR = 1e-3
PATIENCE = 10
LR_FACTOR = 0.8
use_noisy = True
use_augmented = False

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Audio Tagging')
# from https://www.kaggle.com/rio114/keras-cnn-with-lwlrap-evaluation/
def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples, num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true == 1) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)

    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.math.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out

# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class

def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0, 
      scores[nonzero_weight_sample_indices, :],
      sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

class Dataset(Sequence):
    """Creating data generator"""
    def __init__(self, mels, labels):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = iaa.Sequential([
            iaa.CoarseDropout(0.1,size_percent=0.02)
        ])
    
    def getitem(self,image):
        image = Image.fromarray(image, mode='L')        
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = np.array(image)
        image = tf.divide(self.transforms(images=image),255)
        image = np.expand_dims(image,-1)
        image = preprocess_input(image)
        return image
    
    def create_generator(self, batch_size, shuffling=False, test_data=False):
        while True:
            train_X,train_y = self.mels,self.labels
            if shuffling:
                train_X,train_y = shuffle(train_X,train_y)

            for start in range(0, len(train_y), batch_size):
                end = min(start + batch_size, len(train_X))
                batch_data = []
                X_train_batch = train_X[start:end]
                for i in range(len(X_train_batch)):
                    image = self.getitem(X_train_batch[i])
                    batch_data.append(image) 
                if test_data == False:
                    batch_labels = train_y[start:end]
                    
                if test_data == False:
                    yield np.array(batch_data, np.float32), batch_labels.astype('float32') 
                else:
                    yield np.array(batch_data, np.float32)
        return image

def BCEwithLogits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)
def read_audio(conf, pathname, trim_long_data):
    """loading the audio files and applying some effects"""
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)

    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0+conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y


def audio_to_melspectrogram(conf, audio):
    """Converting audio to spectograms"""
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    """print the spectogram image"""
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    """single function for reading and converting the audio to spectrogram"""
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


class conf:
    sampling_rate = 44100
    duration = 2 # sec
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration

def transform(conf, pathname: Path, cmd=[]):
    """applying some audio augmentations with sox"""
    cmd = ["sox", str(pathname), "output.wav"] + cmd
    sp.run(cmd)

    augmented = read_audio(conf, Path("output.wav"), trim_long_data=False)
    mels_augmented = audio_to_melspectrogram(conf, augmented)
    return mels_augmented

def normalize(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    """Normalize and standardize the audio"""
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(file):
    """applying all the transformations above"""
    curated = []
    eq = []
    tb = []
    tbup = []
    reverb = []
    fade = []
    pitchup = []
    pitchdn = []
    augs = [transform(conf, file, cmd=["gain", "-h", "equalizer", "2400", "3q", "8"]),\
    transform(conf, file, cmd=["gain", "-h", "treble", "-30", "bass", "-30"]),\
    transform(conf, file, cmd=["gain", "-h", "treble", "+20", "bass", "+20"]),\
    transform(conf, file, cmd=["reverb"]),\
    transform(conf, file, cmd=["fade", "q", "3"]),\
    transform(conf, file, cmd=["pitch", "+500"]),\
    transform(conf, file, cmd=["pitch", "-500"])]
    for i,j in zip(augs,[eq,tb,tbup,reverb,fade,pitchup,pitchdn]):
        x_color = normalize(i)
        j.append(x_color)
    x = read_as_melspectrogram(conf, file, trim_long_data=False)
    x_color = normalize(x)
    curated.append(x_color)
    return curated,eq,tb,tbup,reverb,fade,pitchup,pitchdn

def load_all_models():
    """Loading all saved ensemble models"""
    # define filename for this ensemble
    filename = '../snapshot_model_8.h5'
    # load model from file
    model = load_model(filename,compile=False)
    model.compile(loss=BCEwithLogits, optimizer=Adam(lr=LR), metrics=[tf_lwlrap])
    print('>loaded %s' % filename)
    return model
 
def ensemble_predictions(model, testX,testY,test=False):
    """make an ensemble prediction for multi-class classification"""
    # make predictions
    validation_generator = Dataset(testX,testY).create_generator(batch_size=BATCH_SIZE,test_data=test)
    pred_val_y = model.predict(validation_generator,steps=len(testX)//BATCH_SIZE+1,verbose=1)
    for i in range(1):
        validation_generator = Dataset(testX,testY).create_generator(batch_size=BATCH_SIZE,test_data=test)
        pred_val_y += model.predict(validation_generator,steps=len(testX)//BATCH_SIZE+1,verbose=1)
    yhats = pred_val_y/1
    return yhats
 
def evaluate_n_members(model, testX, testy,test=False):
    """evaluate a specific number of members in an ensemble"""

    # make prediction
    yhat = ensemble_predictions(model, testX,testy,test=test)
    if test==True:
        return yhat
    # calculate accuracy
    return calculate_overall_lwlrap_sklearn(testy, yhat)

def final(x):
    """Final function 1: takes raw data and source folder and returns predictions"""
    x_train,x_eq,x_tb,x_tbup,x_reverb,x_fade,x_pitchup,x_pitchdn = convert_wav_to_image(x)
    x_aug = x_train + x_fade + x_pitchup + x_reverb + x_tb + x_pitchup + x_eq + x_tbup

    # load models in order
    model = load_all_models()
    y_pred = evaluate_n_members(model, x_aug, x_aug,test=True)
    return np.mean(y_pred,0)

labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 'Bass_guitar',\
 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 'Bus', 'Buzz', 'Car_passing_by', 'Cheering', \
 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping', \
 'Computer_keyboard', 'Crackle', 'Cricket', 'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans',\
  'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',\
   'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock',\
    'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', \
    'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', \
    'Skateboard', 'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise', \
    'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']

audio_bytesio = st.file_uploader("Upload an audio file",type="wav")

if audio_bytesio:
    try:
        with open('myfile.wav', mode='bx') as f:
            f.write(audio_bytesio.getbuffer())
    except FileExistsError:
        os.remove('myfile.wav')
        with open('myfile.wav', mode='bx') as f:
            f.write(audio_bytesio.getbuffer())

    with st.spinner('Inferring output label...'):
        result = final('myfile.wav')
    st.success('Done!')

    label = labels[np.argmax(result)]
    print('Inference done........')
    st.markdown("Events found in the clip are: ")
    st.text(' '.join(label.split('_')))
    audio_bytesio = None
    os.remove('myfile.wav')
    os.remove('output.wav')

if st.button('Show all possible labels'):
    st.write(', '.join([' '.join(i.split('_')) for i in labels]))

