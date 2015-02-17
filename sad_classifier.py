# -*- coding: utf-8 -*-

from blocks.datasets import ContainerDataset
from blocks.datasets.streams import BatchDataStream, PaddingDataStream
from blocks.datasets.schemes import ConstantScheme
from load_SAD import load_SAD

if __name__ == '__main__':
    ########################
    # construct data stream
    ########################
    sad = load_SAD(which_set='train', shuffle=True, seed=0)
    num_examples = len(sad['data'])
    batch_size = 10

    dataset = ContainerDataset(container=sad)
    batch = BatchDataStream(dataset.get_default_stream(), ConstantScheme(batch_size))
    stream = PaddingDataStream(batch)

    # access the data
    it = stream.get_epoch_iterator()
    samples = it.next()
    samples[0] # first batch of samples of 13 dimensional time series data with 0 padded
    samples[1] # masks of above data
