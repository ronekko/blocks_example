# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:17:02 2015

@author: ryuhei
"""
import theano.tensor as tt
from blocks.bricks import Linear, Rectifier, Tanh, Softmax
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.algorithms import GradientDescent, Scale, Adam, Momentum
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

# construct an MLP with 2 layers
x = tt.matrix('features')
input_to_hidden = Linear(name='input_to_hidden',
                         input_dim=784, output_dim=200)
h = Tanh().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output',
                          input_dim=200, output_dim=10)
y_hat = Softmax().apply(hidden_to_output.apply(h))

# construct a cost function with L2 regularization
y = tt.lmatrix('targets')
cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
cg = ComputationGraph(cost)
W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
W1.get_value()
cost = cost + 0.0001 * (W1 ** 2).sum() + 0.0001 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

# initialize the model parameters
input_to_hidden.weights_init = IsotropicGaussian(0.001)
hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = Constant(0)
hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

# setup a training algorithm
#step_rule = Scale(learning_rate=0.01)
#step_rule = Adam()
step_rule = Momentum(learning_rate=0.001, momentum=0.9)
algorithm = GradientDescent(cost=cost,
                            params=cg.parameters,
                            step_rule=step_rule)

# load datasets
mnist = MNIST('train')
data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
    mnist.num_examples, batch_size=256))
mnist_test =MNIST('test')
data_stream_test = DataStream(mnist_test, iteration_scheme=SequentialScheme(
    mnist_test.num_examples, batch_size=1024))

monitor = DataStreamMonitoring(variables=[cost],
                               data_stream=data_stream_test,
                               prefix='test')

main_loop = MainLoop(data_stream=data_stream,
                     algorithm=algorithm,
                     extensions=[monitor,
                                 FinishAfter(after_n_epochs=100),
                                 Printing()]
                    )
main_loop.run()