# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:33:47 2015

@author: ryuhei
"""

#if __name__ == '__main__':
import numpy as np
import theano
import theano.tensor as tt
from blocks import initialization
from blocks.bricks import Identity, Linear
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.cost import SquaredError

seq_len, batch_size, x_dim = (4, 3, 2)
x_val = np.ones((seq_len, batch_size, x_dim))

x = tt.tensor3('x')
rnn = SimpleRecurrent(dim=x_dim, activation=Identity(),
                      weights_init=initialization.Identity())
rnn.initialize()
h = rnn.apply(x)

linear = Linear(input_dim=2, output_dim=1,
                weights_init=initialization.Identity(),
                biases_init=initialization.Constant(0))
linear.initialize()
y_hat = linear.apply(h)


f_h = theano.function([x], h)
print f_h(x_val)

y = tt.tensor3('y')
cost = SquaredError().apply(y, y_hat)

f_cost = theano.function([x, y], cost)
y_val = 4.0 * np.ones((seq_len, batch_size, 1))
print f_cost(x_val, y_val)