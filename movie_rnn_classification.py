# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 00:10:07 2015

@author: ryuhei
"""

import numpy as np
import theano
import theano.tensor as tt
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Padding, Mapping
from blocks.bricks import Linear, Tanh, Rectifier, Softmax
from blocks.bricks.conv import ConvolutionalLayer, Flattener
from blocks.bricks.recurrent import LSTM
from blocks.initialization import Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale, Adam, RMSProp, Momentum
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring
)
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing
import my_datasets

floatX = theano.config.floatX


def swap01(data):
    features, features_mask, targets = data
    return (features.swapaxes(0, 1), features_mask.T, targets)


def reshape_as_non_sequence(data):
    features, features_mask, targets = data
    shape = features.shape
    unrolled_shape = (shape[0]*shape[1], ) + shape[2:]
    return (features.reshape(unrolled_shape), features_mask, targets)

if __name__ == '__main__':
    """
    At this time, since ConvolutionalLayer doesn't support tensor5 for RNN,
    (time*batch, channel, row, col)-shaped tensor4 is fed into conv layer
    first, and then it is folded as (time, batch, channel, row, col) right
    before it is fed into RNN layer.
    """

    # construct train/validation data streams ############################
    image_size = (40, 40)
    batch_size = 100
    # train
    train_size = 500
    assert train_size % batch_size == 0
    train_set = my_datasets.SimpleMovieDataset(train_size, image_size)
    iteration_scheme = ShuffledScheme(train_set.num_examples, batch_size)
    train_stream = DataStream(train_set, iteration_scheme=iteration_scheme)
    train_stream = Padding(train_stream, mask_sources=('features'))
    train_stream = Mapping(train_stream, swap01)
    train_stream = Mapping(train_stream, reshape_as_non_sequence)
    # valid
    valid_size = 200
    assert valid_size % batch_size == 0
    valid_set = my_datasets.SimpleMovieDataset(valid_size, image_size)
    valid_iteration_scheme = SequentialScheme(valid_size, batch_size)
    valid_stream = DataStream(valid_set,
                              iteration_scheme=valid_iteration_scheme)
    valid_stream = Padding(valid_stream, mask_sources=('features'))
    valid_stream = Mapping(valid_stream, swap01)
    valid_stream = Mapping(valid_stream, reshape_as_non_sequence)

    # construct RNN for movie classifier  #############################
#    tensor5 = tt.TensorType(floatX, (False,)*5)
    x_data = tt.tensor4('features')
    x_mask = tt.matrix('features_mask')
    activation = Rectifier().apply
    rnn_dim = 10
    output_dim = 2
    conv1 = ConvolutionalLayer(activation=activation, filter_size=(5, 5),
                               num_filters=10, pooling_size=(4, 4),
                               num_channels=1, image_size=image_size,
                               weights_init=Uniform(
                                   width=np.sqrt(6.0/np.prod(image_size))),
                               biases_init=Constant(0), name='conv1')
    f = Flattener().apply(conv1.apply(x_data))
    f_shape = conv1.get_dim('output')
    f_size = np.prod(f_shape)
    # split 1st axis into 2 axes which are sequence axis and batch axis
    f = f.reshape((-1, batch_size, f_size))
    linear = Linear(name='linear', input_dim=f_size, output_dim=4*rnn_dim,
                    weights_init=Uniform(width=0.1), biases_init=Constant(0.1))
    lstm = LSTM(dim=rnn_dim, activation=Tanh(),
                weights_init=Uniform(width=0.1),
                biases_init=Constant(0), name='lstm')
    linear2 = Linear(name='linear2', input_dim=rnn_dim, output_dim=output_dim,
                     weights_init=Uniform(width=0.2), biases_init=Constant(0))
    h, c = lstm.apply(linear.apply(f), mask=x_mask)
    h_last = h[-1]
    y_hat = Softmax().apply(linear2.apply(h_last))
    conv1.initialize()
    linear.initialize()
    lstm.initialize()
    linear2.initialize()

    # construct the cost function ########################################
    y = tt.lmatrix('targets')
    cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    cg = ComputationGraph(cost)
    cost.name = 'cost'

    # setup the training algorithm #######################################
#    step_rule = Scale(learning_rate=0.01)
#    step_rule = Adam()
#    step_rule = Momentum(learning_rate=0.001, momentum=0.90)
    step_rule = RMSProp(learning_rate=0.001)
    algorithm = GradientDescent(cost=cost,
                                params=cg.parameters,
                                step_rule=step_rule)

    # monitor
    monitor_linear = TrainingDataMonitoring(linear.auxiliary_variables,
                                            after_epoch=True, prefix='linear')
    monitor_linear2 = TrainingDataMonitoring(linear2.auxiliary_variables,
                                             after_epoch=True,
                                             prefix='linear2')
    monitor_train = TrainingDataMonitoring([cost],
                                           after_epoch=True, prefix='train')
    monitor_valid = DataStreamMonitoring(variables=[cost],
                                         data_stream=valid_stream,
                                         prefix='valid')

    # main loop ##########################################################
    main_loop = MainLoop(data_stream=train_stream,
                         algorithm=algorithm,
                         extensions=[FinishAfter(after_n_epochs=10),
                                     monitor_linear, monitor_linear2,
                                     monitor_train, monitor_valid,
                                     Timing(), Printing()])
    main_loop.run()

    # evaluate test data #################################################
    test_size = 1000
    assert test_size % batch_size == 0
    test_set = my_datasets.SimpleMovieDataset(test_size, image_size)
    test_iteration_scheme = SequentialScheme(test_size, batch_size)
    test_stream = DataStream(test_set,
                             iteration_scheme=test_iteration_scheme)
    test_stream = Padding(test_stream, mask_sources=('features'))
    test_stream = Mapping(test_stream, swap01)
    test_stream = Mapping(test_stream, reshape_as_non_sequence)

    count_correct = tt.sum(tt.eq(y.flatten(), tt.argmax(y_hat, axis=1)))
    func = theano.function([x_data, x_mask, y], count_correct)
    num_correct = 0
    for test_batch in test_stream.get_epoch_iterator():
        test_data, test_mask, test_target = test_batch
        num_correct += func(test_data, test_mask, test_target)
    accuracy = num_correct / float(test_size)
    print "accuracy = %.3f (%d/%d)" % (accuracy, num_correct, test_size)
