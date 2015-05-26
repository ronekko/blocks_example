# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:17:02 2015

@author: ryuhei
"""
import numpy as np
import theano
import theano.tensor as tt
from blocks.bricks import MLP, Identity, Linear, Rectifier, Softmax
from blocks.bricks.conv import (
    ConvolutionalLayer, ConvolutionalSequence, Flattener
)
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import INPUT, DROPOUT
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping
from blocks.algorithms import GradientDescent, Adam, Momentum
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

linear = MLP([Identity(), Identity()], [2, 10, 2],
              weights_init=Constant(1), biases_init=Constant(2))
x = tt.matrix('x')
y = linear.apply(x)
linear.initialize()

cg = ComputationGraph(y)

inputs = VariableFilter(roles=[INPUT])(cg.variables)
cg_dropout = apply_dropout(cg, inputs, 0.5)
dropped_out = VariableFilter(roles=[DROPOUT])(cg_dropout.variables)

inputs_referenced = [var.tag.replacement_of for var in dropped_out]

fprop = theano.function(cg.inputs, cg.outputs[0])
print fprop(np.ones((3, 2), dtype=theano.config.floatX))

fprop_dropout = theano.function(cg_dropout.inputs, cg_dropout.outputs[0])
print fprop_dropout(np.ones((3, 2), dtype=theano.config.floatX))
print fprop_dropout(np.ones((3, 2), dtype=theano.config.floatX))
print fprop_dropout(np.ones((3, 2), dtype=theano.config.floatX))
