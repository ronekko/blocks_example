# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:59:56 2015

@author: ryuhei
"""

from collections import OrderedDict
import numpy as np
import theano
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

floatX = theano.config.floatX


class Sequence1dDataset(IndexableDataset):
    def __init__(self, num_examples=1000):
        data = self.generate_sequential_data(num_examples)
        sources = ('features', 'targets')
        indexables = OrderedDict(zip(sources, data))
        super(Sequence1dDataset, self).__init__(indexables)

    def generate_sequential_data(self, num_examples=1000):
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

if __name__ == '__main__':
    dataset = Sequence1dDataset(15)
    batch_size = 10
    iteration_scheme = SequentialScheme(dataset.num_examples, batch_size)
    stream = DataStream(dataset, iteration_scheme=iteration_scheme)
    epoch = stream.get_epoch_iterator()
    batch = epoch.next()
