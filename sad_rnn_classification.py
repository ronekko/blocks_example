# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 15:40:07 2015

@author: ryuhei
"""

import theano
import theano.tensor as tt
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Padding, Mapping
from blocks.bricks import Linear, Tanh, Softmax
from blocks.bricks.recurrent import LSTM
from blocks.initialization import Uniform, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale, Adam, RMSProp, Momentum
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Timing, Printing
import my_datasets

floatX = theano.config.floatX


def swap01(data):
    features, features_mask, targets = data
    return (features.swapaxes(0, 1), features_mask.T, targets)

if __name__ == '__main__':
    # construct train/validation data streams ############################
    # train
    train_set = my_datasets.SADDataset('train')
    batch_size = 400
    iteration_scheme = ShuffledScheme(train_set.num_examples, batch_size)
    train_stream = DataStream(train_set, iteration_scheme=iteration_scheme)
    train_stream = Padding(train_stream, mask_sources=('features'))
    train_stream = Mapping(train_stream, swap01)
    # test
    test_set = my_datasets.SADDataset('test')
    test_size = test_set.num_examples
    test_iteration_scheme = SequentialScheme(test_size, test_size)
    test_stream = DataStream(test_set,
                             iteration_scheme=test_iteration_scheme)
    test_stream = Padding(test_stream, mask_sources=('features'))
    test_stream = Mapping(test_stream, swap01)

    # construct RNN for sequence classifier  #############################
    x_data = tt.tensor3('features')
    x_mask = tt.matrix('features_mask')
    input_dim = 13
    rnn_dim = 25
    output_dim = 10
    linear = Linear(name='linear', input_dim=input_dim, output_dim=4*rnn_dim,
                    weights_init=Uniform(width=0.02), biases_init=Constant(0))
    lstm = LSTM(dim=rnn_dim, activation=Tanh(),
                weights_init=Uniform(width=0.3),
                biases_init=Constant(0))
    linear2 = Linear(name='linear2', input_dim=rnn_dim, output_dim=output_dim,
                     weights_init=Uniform(width=0.2), biases_init=Constant(0))
    h, c = lstm.apply(linear.apply(x_data), mask=x_mask)
    h_last = h[-1]
    y_hat = Softmax().apply(linear2.apply(h_last))

    # construct the cost function ########################################
    y = tt.lmatrix('targets')
    cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    cg = ComputationGraph(cost)
    cost.name = 'cost'

    # initialize the model parameters ####################################
    linear.initialize()
    lstm.initialize()
    linear2.initialize()

    # setup the training algorithm #######################################
#    step_rule = Scale(learning_rate=0.01)
#    step_rule = Adam()
#    step_rule = Momentum(learning_rate=0.001, momentum=0.90)
    step_rule = RMSProp(learning_rate=0.01)
    algorithm = GradientDescent(cost=cost,
                                params=cg.parameters,
                                step_rule=step_rule)

    # monitor
    monitor = DataStreamMonitoring(variables=[cost],
                                   data_stream=test_stream,
                                   prefix='test')

    # main loop ##########################################################
    main_loop = MainLoop(data_stream=train_stream,
                         algorithm=algorithm,
                         extensions=[FinishAfter(after_n_epochs=50),
                                     monitor,
                                     Timing(),
                                     Printing()])
    main_loop.run()

    # evaluate test data #################################################
    test_batch = test_stream.get_epoch_iterator().next()
    test_data, test_mask, test_target = test_batch
    count_correct = tt.sum(tt.eq(y.flatten(), tt.argmax(y_hat, axis=1)))
    f = theano.function([x_data, x_mask, y], count_correct)
    num_correct = f(test_data, test_mask, test_target)
    accuracy = num_correct / float(test_size)
    print "accuracy = %.3f (%d/%d)" % (accuracy, num_correct, test_size)
