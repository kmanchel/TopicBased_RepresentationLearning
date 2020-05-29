"""This script creates a tf.data.Dataset data loader pipeline"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

sys.path.append("..")
sys.path

import tensorflow as tf
import pandas as pd
import numpy as np
import string
import tensorflow_datasets as tfds
from utils import Params

def get_dataset(file_path,batch_size, **kwargs):
    """Data loader from dataset csv file
        Args:
            params: (Params) Model Parameters (dataset path is used here)
        Return:
            dataset: (tf.tfds.data.make_csv_dataset) 
    """
    
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,select_columns=['content','bias'],batch_size=batch_size,label_name='bias',
        ignore_errors=True,shuffle=True, **kwargs)
    return dataset


def fix_fn(features, label):
    """Transforms (tf.tfds.data.make_csv_dataset) to (tf.tfds.data.dataset) 
        Args:
            features: (oDict) From csv dataset object
            labels: (tf.int32) From csv dataset object
        Return:
            features: (tf.string) Basically maps this from oDict to string
            label: (tf.int32)
    """
    def _fix(text,label):
        text = text.numpy()
        return text,label
    
    return tf.py_function(_fix, inp=[features['text'], label], Tout=(tf.string, tf.int32))


class Corpus:
    """Just a class to formalize the dataloader
        Args:
            params: (Params) Model Parameters (dataset paths are needed here) 
    """
    def __init__(self,params):
        print("Loaded Training Articles from %s"%params.clean_dataset_train)
        self.train = get_dataset(params.clean_dataset_train,params.batch_size).map(fix_fn)
        print("Loaded Validation Articles from %s"%params.clean_dataset_val)
        self.val = get_dataset(params.clean_dataset_val,params.batch_size).map(fix_fn)