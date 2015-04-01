# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:57:37 2015

@author: ryuhei
"""

from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme


if __name__ == '__main__':
    mnist = MNIST(which_set='train')
    stream = DataStream(mnist,
                        iteration_scheme=ShuffledScheme(mnist.num_examples,
                                                        512))
    epoch = stream.get_epoch_iterator()
    batch = epoch.next() # (features, targets)
    features, targets = batch
    print features.shape, targets.shape # (512L, 784L) (512L, 1L)
    feature0_1st = features[0]

    for i, batch in enumerate(epoch):
        features, targets = batch
        print "%03d: "%i, len(features)

    epoch = stream.get_epoch_iterator()
    batch = epoch.next()
    feature0_2nd = batch[0][0]

    epoch = stream.get_epoch_iterator()
    for i, batch in enumerate(epoch):
        features, targets = batch
        print "%03d: "%i, len(features)

    # different epoch iterators generate batches in different order
    print (feature0_1st == feature0_2nd).all() # False