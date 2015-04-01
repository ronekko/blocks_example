# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:09:20 2015

@author: ryuhei
"""

from fuel.datasets.hdf5 import H5PYDataset

if __name__ == '__main__':
    """ Construct H5PYDataset from HDF5 file.

    Make sure the file "dataset.hdf5" exists in the same directory
    before run this script. If not, run "create_hdf5.py" beforehand.
    """

    dataset_name = "dataset.hdf5"

    # load entire dataset
    dataset = H5PYDataset(dataset_name)
    print "dataset.num_examples = %d" % dataset.num_examples # 100

    # load train/test set with which_set
    train_set = H5PYDataset(dataset_name, which_set="train")
    test_set  = H5PYDataset(dataset_name, which_set="test")
    print "train_set.num_examples = %d" % train_set.num_examples # 90
    print "test_set.num_examples = %d" % test_set.num_examples   # 10

    # divide train set into train/validation set with subset
    train_set = H5PYDataset(dataset_name, which_set="train",
                            subset=slice(0, 80))
    valid_set = H5PYDataset(dataset_name, which_set="train",
                            subset=slice(80, 90))
    print "train_set.num_examples = %d" % train_set.num_examples # 80
    print "valid_set.num_examples = %d" % valid_set.num_examples # 10

    print "train_set.provides_sources = \n\t", train_set.provides_sources

    handle = train_set.open()
    data = train_set.get_data(handle, slice(0, 10))
    print "data[0].shape, data[1].shape, data[2].shape = \n\t", \
        data[0].shape, data[1].shape, data[2].shape
    train_set.close(handle)

    # We can also request just the vector features
    train_vector_features = H5PYDataset(dataset_name, which_set="train",
                                        subset=slice(0, 80),
                                        sources=["vector_features"])
    handle = train_vector_features.open()
    data, = train_vector_features.get_data(handle, slice(0, 20))
    print "data.shape = ", data.shape # (10, 20)
    train_vector_features.close(handle)

    # Loading data in memory
    in_memory_train_vector_features = H5PYDataset(
        dataset_name, which_set="train", subset=slice(0, 80),
        sources=["vector_features"], load_in_memory=True)
    data, = in_memory_train_vector_features.data_sources
    print "type(data) = ", type(data)
    print "data.shape = ", data.shape