# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:39:35 2015

@author: ryuhei
"""

import os
import numpy as np
import h5py

def make_data_files(remake=False):
    if remake or not os.path.exists('train_vector_features.npy'):
        np.save('train_vector_features.npy',
                np.random.normal(size=(90, 10)).astype(np.float32))
        np.save('test_vector_features.npy',
                np.random.normal(size=(10, 10)).astype(np.float32))
        np.save('train_image_features.npy',
                np.random.randint(2, size=(90, 3, 5, 5)).astype(np.uint8))
        np.save('test_image_features.npy',
                np.random.randint(2, size=(10, 3, 5, 5)).astype(np.uint8))
        np.save('train_targets.npy',
                np.random.randint(10, size=(90, 1)).astype(np.uint8))
        np.save('test_targets.npy',
                np.random.randint(10, size=(10, 1)).astype(np.uint8))

if __name__ == '__main__':
    make_data_files(remake=False)

    train_vector_features = np.load('train_vector_features.npy')
    test_vector_features = np.load('test_vector_features.npy')
    train_image_features = np.load('train_image_features.npy')
    test_image_features = np.load('test_image_features.npy')
    train_targets = np.load('train_targets.npy')
    test_targets = np.load('test_targets.npy')

    f = h5py.File('dataset.hdf5', mode='w')
    vector_features = f.create_dataset('vector_features',
                                       (100, 10), dtype=np.float32)
    image_features = f.create_dataset('image_features',
                                      (100, 3, 5, 5), dtype=np.uint8)
    targets = f.create_dataset('targets', (100, 1), dtype=np.uint8)

    vector_features[...] = np.vstack([train_vector_features,
                                      test_vector_features])
    image_features[...] = np.vstack([train_image_features,test_image_features])
    targets[...] = np.vstack([train_targets, test_targets])

    f.attrs['train'] = [0,90]
    f.attrs['test'] = [90,100]

    f.flush()
    f.close()