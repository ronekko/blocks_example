# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:59:56 2015

@author: ryuhei
"""

import os
from collections import OrderedDict
from StringIO import StringIO
import numpy as np
import theano
from fuel import config
from fuel.datasets import IndexableDataset

floatX = theano.config.floatX


class Sequence1dDataset(IndexableDataset):
    def __init__(self, num_examples=1000):
        data = Sequence1dDataset.generate_sequential_data(num_examples)
        sources = ('features', 'targets')
        indexables = OrderedDict(zip(sources, data))
        super(Sequence1dDataset, self).__init__(indexables)

    @staticmethod
    def generate_sequential_data(num_examples=1000):
        noise_std = 1
        lengths = 15 + np.random.poisson(5, num_examples)
        # generate targets
        targets = np.zeros((num_examples, 1), dtype=np.int64)
        targets[:num_examples/2] = 1
        np.random.shuffle(targets)
        # generate features
        features = []
        for k, length in zip(targets, lengths):
            sign = 1.0 if k else -1.0
            x = np.linspace(0, 2*np.pi, length)
            y = sign * np.sin(x) + noise_std * np.random.randn(length)
            features.append(y.astype(floatX).reshape(length,1))

        return (np.array(features), targets)


class SADDataset(IndexableDataset):
    """ "Spoken Arabic Digits" dataset in Fuel format

    Parameter
    --------------
    which_set: 'train' or 'test'

    Data Source
    --------------
    You can download the dataset file from
    https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit
    """
    def __init__(self, which_set):
        indexables = SADDataset.load_SAD(which_set)
        super(SADDataset, self).__init__(indexables)

    @staticmethod
    def load_SAD(which_set, shuffle=False, seed=None):
        """ load SAD dataset

        which_set: 'train' or 'test'
        """
        data_file = dict(
            train = 'Train_Arabic_Digit.txt',
            test = 'Test_Arabic_Digit.txt'
        )
        assert which_set in data_file.keys(), (
            "`which_set` must be 'train' or 'test'.")

        dir_path = os.path.join(config.data_path, "SAD")
        dataset_path = os.path.join(dir_path, data_file[which_set])

        with open(dataset_path) as f:
            data = []
            for block in f.read().strip().split('\n            \n'):
                sio = StringIO(block)
                array = np.loadtxt(sio, dtype=np.float32)
                data.append(array)

            data_len = len(data)
            data = np.array(data)
            target_list = []
            for i in range(10):
                target_list = target_list + [i] * (data_len / 10)
            targets = np.array(target_list).reshape(data_len, 1)

        if shuffle:
            # zip -> shuffle -> unzip
            list_of_pair = list(zip(data, targets))
            random_state = np.random.RandomState(seed)
            random_state.shuffle(list_of_pair)
            data, targets = zip(*list_of_pair)
            data = np.array(data)
            targets = np.array(targets)

        return OrderedDict((("features", data), ("targets", targets)))

if __name__ == '__main__':
    seq = Sequence1dDataset(2200)
    seq_data = seq.get_data(request=slice(0,seq.num_examples))
    print "# Sequence1dDataset ###"
    print "type(seq_data) =", type(seq_data)
    print "len(seq_data) =", len(seq_data)
    print "type(seq_data[0]) =", type(seq_data[0])
    print "seq_data[0].shape =", seq_data[0].shape
    print "seq_data[0][0].shape =", seq_data[0][0].shape
    print "seq_data[1][0].shape =", seq_data[1][0].shape
    print ""

    sad = SADDataset('test')
    sad_data = sad.get_data(request=slice(0,sad.num_examples))
    print "# SADDataset ###"
    print "type(sad_data) =", type(sad_data)
    print "len(sad_data) =", len(sad_data)
    print "type(sad_data[0]) =", type(sad_data[0])
    print "sad_data[0].shape =", sad_data[0].shape
    print "sad_data[0][0].shape =", sad_data[0][0].shape
    print "sad_data[1][0].shape =", sad_data[1][0].shape
    print ""

    sad_data = SADDataset.load_SAD('test')
    print "# SADDataset.load_SAD ###"
    print "type(sad_data) =", type(sad_data)
    print "len(sad_data) =", len(sad_data)
    print "type(sad_data['features']) =", type(sad_data['features'])
    print "sad_data['features'].shape =", sad_data['features'].shape
    print "sad_data['features'][0].shape =", sad_data['features'][0].shape
    print "sad_data['targets'][0].shape =", sad_data['targets'][0].shape
    print ""
