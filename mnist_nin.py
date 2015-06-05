# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:17:02 2015

@author: ryuhei
"""
import numpy as np
import theano
import theano.tensor as tt
from theano.sandbox.cuda.dnn import dnn_pool
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.conv import (
    ConvolutionalLayer, ConvolutionalActivation, MaxPooling,
    ConvolutionalSequence, Flattener
)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import INPUT, WEIGHT
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping
from blocks.algorithms import GradientDescent, Adam, RMSProp, Momentum
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring
)
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, Timing

# construct a ConvNet with 2 conv+maxpool+ReLU components and softmax
x = tt.tensor4('features')
activation = Rectifier().apply
mlpconv11 = ConvolutionalActivation(activation=activation, filter_size=(5, 5),
                                    num_filters=100,
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0), name='mlpconv11')
mlpconv12 = ConvolutionalActivation(activation=activation, filter_size=(1, 1),
                                    num_filters=100,
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0), name='mlpconv12')
mlpconv13 = ConvolutionalLayer(activation=activation, filter_size=(1, 1),
                               num_filters=80, pooling_size=(2, 2),
                               weights_init=IsotropicGaussian(0.01),
                               biases_init=Constant(0), name='mlpconv13')

mlpconv21 = ConvolutionalActivation(activation=activation, filter_size=(5, 5),
                                    num_filters=100,
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0), name='mlpconv21')
mlpconv22 = ConvolutionalActivation(activation=activation, filter_size=(1, 1),
                                    num_filters=100,
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0), name='mlpconv22')
mlpconv23 = ConvolutionalActivation(activation=activation, filter_size=(1, 1),
                                    num_filters=10,
                                    weights_init=IsotropicGaussian(0.1),
                                    biases_init=Constant(0), name='mlpconv23')

conv_seq = ConvolutionalSequence([mlpconv11, mlpconv12, mlpconv13,
                                  mlpconv21, mlpconv22, mlpconv23],
                                 num_channels=1, image_size=(28, 28))
# conv_seq.push_allocation_config()
f = conv_seq.apply(x)
conv_seq.initialize()
tensor_shapes = [child.get_dim('input_') for child in conv_seq.children] + [
    conv_seq.get_dim('output')]

# global average pooling
f = dnn_pool(f, tensor_shapes[-1][-2:], mode='average')

h = Flattener().apply(f)
y_hat = Softmax().apply(h)

# construct a cost function with L2 regularization
y = tt.lmatrix('targets')
cost_misclass = MisclassificationRate().apply(y.flatten(), y_hat)
cost_misclass.name = 'cost_missclass'
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
cost.name = 'cost_without_regularization'
cg = ComputationGraph(cost)

# configure dropout
dropout_bricks = [mlpconv23]
variable_filter = VariableFilter([INPUT], dropout_bricks)
inputs = variable_filter(cg.variables)
cg = apply_dropout(cg, inputs, 0.5)
cost = cg.outputs[0]

# L2 regularization
weights = VariableFilter([WEIGHT])(cg.variables)
# cost += 0.00001 * sum([(W ** 2).sum() for W in weights])
# cost.name = 'cost_with_regularization'

# setup a training algorithm
#step_rule = Adam(0.01)
step_rule = RMSProp(0.01)
# step_rule = Momentum(learning_rate=0.001, momentum=0.9)
algorithm = GradientDescent(cost=cost,
                            params=cg.parameters,
                            step_rule=step_rule)


# load datasets
def vector2image(data):
    return (data[0].reshape(-1, 1, 28, 28), data[1])

mnist = MNIST('train')
data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
    mnist.num_examples, batch_size=200))
data_stream = Mapping(data_stream, vector2image)
mnist_test = MNIST('test')
data_stream_test = DataStream(mnist_test, iteration_scheme=SequentialScheme(
    mnist_test.num_examples, batch_size=1000))
data_stream_test = Mapping(data_stream_test, vector2image)

monitor_train = TrainingDataMonitoring([cost, cost_misclass],
                                       after_epoch=True, prefix='train')
monitor_test = DataStreamMonitoring(variables=[cost, cost_misclass],
                                    data_stream=data_stream_test,
                                    prefix='test')

main_loop = MainLoop(data_stream=data_stream,
                     algorithm=algorithm,
                     extensions=[monitor_train, monitor_test,
                                 FinishAfter(after_n_epochs=250),
                                 Timing(), Printing()]
                    )
main_loop.run()

# evaluate test data #################################################
count_correct = tt.sum(tt.eq(y.flatten(), tt.argmax(y_hat, axis=1)))
f = theano.function([x, y], count_correct)
test_size = mnist_test.num_examples
num_correct = 0
for test_batch in data_stream_test.get_epoch_iterator():
    test_data, test_target = test_batch
    num_correct += f(test_data, test_target)
accuracy = num_correct / float(test_size)
print "accuracy = %.3f (%d/%d)" % (accuracy, num_correct, test_size)
