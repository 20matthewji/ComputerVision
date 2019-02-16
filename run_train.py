
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

classes = 2
learning_rate=0.1
chnl_size=1;
nkerns=[20,50]
train_batch_size=classes
test_batch_size=classes

dataFile='nnet_SHgray.pkl.gz'
f=gzip.open(dataFile,'rb')
dataset = cPickle.load(f)
f.close()

train_set_x, train_set_y = dataset['train']
valid_set_x, valid_set_y = dataset['valid']
test_set_x, test_set_y = dataset['test']
train_set_x =(train_set_x-np.mean(train_set_x))/np.std(train_set_x)
valid_set_x =(valid_set_x-np.mean(valid_set_x))/np.std(valid_set_x)
test_set_x =(test_set_x-np.mean(test_set_x))/np.std(test_set_x)
n_train_batches = train_set_y.shape[0]/train_batch_size
n_valid_batches = valid_set_y.shape[0]/test_batch_size
n_test_batches = test_set_y.shape[0]/test_batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

print('... building the model')
# rng = np.random.RandomState(23455)
rng = np.random.RandomState(1234)

layer0_input = x.reshape((train_batch_size, chnl_size, 100, 100))
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(train_batch_size, chnl_size, 100, 100),
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


layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 19 * 19,
    n_out=500,
    activation=T.nnet.softmax
)
## option: activation=T.nnet.relu, T.nnet.softmax, tanh

layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=classes)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
sh_test_x=theano.shared(test_set_x)
sh_test_y=theano.shared(test_set_y)
test_model = theano.function(
    [index],
    [layer3.y_pred, layer3.errors(y)],
    givens={
        x: sh_test_x[index * test_batch_size: (index + 1) * test_batch_size],
        y: sh_test_y[index * test_batch_size: (index + 1) * test_batch_size]
    }
)

sh_valid_x=theano.shared(valid_set_x)
sh_valid_y=theano.shared(valid_set_y)
validate_model = theano.function(
    [index],
    layer3.errors(y),
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
n_epochs = 50

print('... training')
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        minibatch_cost_ij = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print('training @ iter = ', iter)
        print('minibatch @ iter = %s \/ %s, costs: %s'% (iter, minibatch_index, minibatch_cost_ij))

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                for i in range(n_test_batches): 
 		    test_results=test_model(i) 
                test_pred = test_results[0]
                test_losses = test_results[1]
                test_score = np.mean(test_losses)
                # print(('     epoch %i, minibatch %i/%i, test error of ' 'best model %d %d %d') %
                #      (epoch, minibatch_index + 1, n_train_batches, test_losses[0][0], test_losses[0][1], test_losses[0][2]))
                print(('     epoch %i, minibatch %i/%i, test pred of \[%i  %i\] error of ' 'best model %f %%') %
                (epoch, minibatch_index + 1, n_train_batches, test_pred[0], test_pred[1], test_score * 100.))
                if ((test_score==0.0) & (this_validation_loss==0.0)):
		   end_time=timeit.default_timer()
                   break
        if patience <= iter:
            done_looping = True
            break

# end_time = timeit.default_timer()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print(('The code for file ' +
       ' ran for %.2fs' % ((end_time - start_time) / 1.0)))
#  file=sys.stderr)
#        os.path.split(__file__)[1] +

