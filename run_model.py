
import os
import sys
import timeit
import cPickle, gzip

import numpy as np
import mahotas as mh

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from logistic_sgd import LogisticRegression, load_data
from convPoolLayer import LeNetConvPoolLayer
from hiddenLayer import HiddenLayer

from PIL import Image
from PIL import ImageFilter
from random import randint
from matplotlib import pyplot as plt

learning_rate=0.1
nkerns=[20,50]
train_batch_size=3
test_batch_size=3
chnl_size=1;

dataFile = 'nnet_SOHgredge.pkl.gz'
f=gzip.open(dataFile,'rb')
dataset = cPickle.load(f)
f.close()

train_set_x, train_set_y = dataset['train']
valid_set_x, valid_set_y = dataset['valid']
test_set_x, test_set_y = dataset['test']
train_set_x =(train_set_x-np.mean(train_set_x))/np.std(train_set_x)
valid_set_x =(valid_set_x-np.mean(valid_set_x))/np.std(valid_set_x)
test_set_x =(test_set_x-np.mean(test_set_x))/np.std(test_set_x)

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

print('... building the model')
rng = np.random.RandomState(23455)

layer0_input = x.reshape((train_batch_size, chnl_size, 100, 100))
layer0_test_input = x.reshape((test_batch_size, chnl_size, 100, 100))
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(train_batch_size, chnl_size, 100, 100),
    filter_shape=(nkerns[0], chnl_size, 9, 9),
    poolsize=(2, 2)
)

layer0tst = LeNetConvPoolLayer(
    rng,
    input=layer0_test_input,
    image_shape=(test_batch_size, chnl_size, 100, 100),
    filter_shape=(nkerns[0], chnl_size, 9, 9),
    poolsize=(2, 2)
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(train_batch_size, nkerns[0], 46, 46),
    filter_shape=(nkerns[1], nkerns[0], 9, 9),
    poolsize=(2, 2)
)

layer1tst = LeNetConvPoolLayer(
    rng,
    input=layer0tst.output,
    image_shape=(test_batch_size, nkerns[0], 46, 46),
    filter_shape=(nkerns[1], nkerns[0], 9, 9),
    poolsize=(2, 2)
)

layer2_input = layer1.output.flatten(2)
layer2_test_input = layer1tst.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 19 * 19,
    n_out=500,
    activation=T.nnet.softmax
)
## option: activation=T.nnet.relu, tanh
layer2tst = HiddenLayer(
    rng,
    input=layer2_test_input,
    n_in=nkerns[1] * 19 * 19,
    n_out=500,
    activation=T.nnet.softmax
)

layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=3)
layer3tst = LogisticRegression(input=layer2tst.output, n_in=500, n_out=3)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
sh_test_x=theano.shared(test_set_x)
sh_test_y=theano.shared(test_set_y)
test_model = theano.function(
    [index],
    layer3tst.errors(y),
    givens={
        x: sh_test_x[index * test_batch_size: (index + 1) * test_batch_size],
        y: sh_test_y[index * test_batch_size: (index + 1) * test_batch_size]
    }
)

sh_valid_x=theano.shared(valid_set_x)
sh_valid_y=theano.shared(valid_set_y)
validate_model = theano.function(
    [index],
    layer3tst.errors(y),
    givens={
        x: sh_valid_x[index * test_batch_size: (index + 1) * test_batch_size],
        y: sh_valid_y[index * test_batch_size: (index + 1) * test_batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

shared_x=theano.shared(train_set_x)
shared_y=theano.shared(train_set_y)
train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: shared_x[index * train_batch_size: (index + 1) * train_batch_size],
        y: shared_y[index * train_batch_size: (index + 1) * train_batch_size]
    }
)

    ###############
    # TRAIN MODEL #
    ###############
n_train_batches = 8
n_valid_batches = 1
n_test_batches = 1
n_epochs = 200

print('... training')
# early-stopping parameters
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

## SH: iter#224 time=0.38ms
## OH: iter#320 time=0.55ms
## SO: iter#184 time=0.32ms
## SOH: iter#312 time=0.54ms --- test score 33.33% (1/3 mismatch)
