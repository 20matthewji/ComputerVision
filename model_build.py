#!/usr/bin/python

import os
import sys
import timeit
import cPickle, gzip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from convPoolLayer import LeNetConvPoolLayer
from hiddenLayer import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


## Start of main
def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mt_data.pkl.gz',
                    nkerns=[20, 50], train_batch_size=16, test_batch_size=2):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

  #  datasets = load_data(dataset)
    f=gzip.open('my_data.pkl.gz','rb')
    dataset = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = dataset['train']
    valid_set_x, valid_set_y = dataset['valid']
    test_set_x, test_set_y = dataset['test']
    train_set_x =(train_set_x-np.mean(train_set_x))/np.std(train_set_x)
    valid_set_x =(valid_set_x-np.mean(valid_set_x))/np.std(valid_set_x)
    test_set_x =(test_set_x-np.mean(test_set_x))/np.std(test_set_x)

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    # n_train_batches //= train_batch_size
    # n_valid_batches //= test_batch_size
    # n_test_batches //= test_batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    rng = numpy.random.RandomState(23455)

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (height, width) is the size of MNIST images.
    # layer0_input = x.reshape((batch_size, 3, 115, 128))
    layer0_input = x.reshape((train_batch_size, 3, 100, 100))
    layer0_test_input = x.reshape((test_batch_size, 3, 100, 100))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (115-26+1 , 128-29+1) = (90, 100)
    # filtering reduces the image size to (100-21+1 , 100-21+1) = (80, 80)
    # maxpooling reduces this further to (90/2, 100/2) = (45, 50)
    # maxpooling reduces this further to (80/2, 80/2) = (40, 40)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 40, 40)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(train_batch_size, 3, 100, 100),
        filter_shape=(nkerns[0], 3, 21, 21),
        poolsize=(2, 2)
    )

    layer0tst = LeNetConvPoolLayer(
        rng,
        input=layer0_test_input,
        image_shape=(test_batch_size, 3, 100, 100),
        filter_shape=(nkerns[0], 3, 21, 21),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (45-26+1, 50-29+1) = (20, 22)
    # filtering reduces the image size to (40-21+1, 40-21+1) = (20, 20)
    # maxpooling reduces this further to (20/2, 22/2) = (10, 11)
    # maxpooling reduces this further to (20/2, 20/2) = (10, 10)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 10, 10)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(train_batch_size, nkerns[0], 40, 40),
        filter_shape=(nkerns[1], nkerns[0], 21, 21),
        poolsize=(2, 2)
    )

    layer1tst = LeNetConvPoolLayer(
        rng,
        input=layer0tst.output,
        image_shape=(test_batch_size, nkerns[0], 40, 40),
        filter_shape=(nkerns[1], nkerns[0], 21, 21),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 10 * 11),
    # or (8, 50 * 10 * 11) = (8, 5500) with the default values.
    # or (8, 50 * 10 * 10) = (8, 5000) with the default values.
    layer2_input = layer1.output.flatten(2)
    layer2_test_input = layer1tst.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 10 * 10,
        n_out=500,
        activation=T.nnet.relu
    )
## option: activation=T.nnet.relu, tanh
    layer2tst = HiddenLayer(
        rng,
        input=layer2_test_input,
        n_in=nkerns[1] * 10 * 10,
        n_out=500,
        activation=T.nnet.relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)
    layer3tst = LogisticRegression(input=layer2tst.output, n_in=500, n_out=2)

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
  # end-snippet-1


    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
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
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of ' 'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

#    open('layer0_model.pkl', 'wb') as f0:
#    pickle.dump(layer0,f0)
#    open('layer1_model.pkl', 'wb') as f1:
#    pickle.dump(layer1,f1)
#    open('layer2_model.pkl', 'wb') as f2:
#    pickle.dump(layer0,f2)
#    open('layer3_model.pkl', 'wb') as f3:
#    pickle.dump(layer0,f3)

#    layer0=pickle.load(open('layer0_model.pkl'))
#    layer1=pickle.load(open('layer1_model.pkl'))
#    layer2=pickle.load(open('layer2_model.pkl'))
#    layer3=pickle.load(open('layer3_model.pkl'))

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    import theano.tensor as T
    import cPickle, numpy
    my_x = np.array([]);
    dir='/Users/Matthew/Documents/workspace/SRC/Strt/'
    for name in ('Asbury_Ave', 'Mendocino_Ln', 'Monterey_Ct', 'Pacific_Ave', 'Park_Ave', 'Roosevelt_Ave', 'Terra_Bella_Dr', 'Terrace_Dr'):
        im=Image.open(dir+name+".edge.png")
        imarr = np.asarray(im)/256.
        height,width=imarr.shape[:2]
        imarrML = np.transpose(imarr,(2,0,1))	# same as imarr.swapaxes(0,-1) then (1, -1)
        imarrML = np.append(imarrML.reshape(3*height*width),0)
        imarrML = imarrML[np.newaxis,:]
        if (my_x.size==0):
           my_x = imarrML
        else:
           my_x = np.concatenate((my_x, imarrML),axis=0)

    dir='/Users/Matthew/Documents/workspace/SRC/Obs/'
    label = np.ones([1], dtype='int32')
    for name in ('Newport_tree', 'Castleton_myrtle', 'Castleton_palm', 'Palm_post', 'Wacker_obs', 'Fairview_post', 'Washington_post', 'Bubb_Redwood'):
        im=Image.open(dir+name+".thumb.png")
        imarr = np.asarray(im)/256.
        height,width=imarr.shape[:2]
        imarrML = np.transpose(imarr,(2,0,1))	# same as imarr.swapaxes(0,-1) then (1, -1)
        imarrML = np.append(imarrML.reshape(3*height*width),1)
        imarrML = imarrML[np.newaxis,:]
        if (my_x.size==0):
           my_x = imarrML
        else:
           my_x = np.concatenate((my_x, imarrML),axis=0)
    np.random.shuffle(my_x)
    row=my_x.shape[0]
    col=my_x.shape[1]-1
    x2=np.float32(my_x[:,:col].reshape((row,col)))
    my_y=np.int32(my_x[:,col:].reshape(row))
    my_x=x2

    dir='/Users/Matthew/Documents/workspace/SRC/Strt/'
    name="Regan_Ln"
    im=Image.open(dir+name+".edge.png")
    imarr = np.asarray(im)/256.
    height,width=imarr.shape[:2]
    imarrML = np.transpose(imarr,(2,0,1))
    imarrML = imarrML.reshape(3*height*width)
    val_x = imarrML[np.newaxis,:]
    name="Dumas_Dr"
    im=Image.open(dir+name+".edge.png")
    imarr = np.asarray(im)/256.
    height,width=imarr.shape[:2]
    imarrML = np.transpose(imarr,(2,0,1))
    imarrML = imarrML.reshape(3*height*width)
    test_x = imarrML[np.newaxis,:]

    dir='/Users/Matthew/Documents/workspace/SRC/Obs/'
    name="Harker_Obs"
    im=Image.open(dir+name+".edge.png")
    imarr = np.asarray(im)/256.
    height,width=imarr.shape[:2]
    imarrML = np.transpose(imarr,(2,0,1))
    imarrML = imarrML.reshape(3*height*width)
    imarrML = imarrML[np.newaxis,:]
    val_x = np.concatenate((val_x, imarrML),axis=0)
    name="Dumas_fyard"
    im=Image.open(dir+name+".edge.png")
    imarr = np.asarray(im)/256.
    height,width=imarr.shape[:2]
    imarrML = np.transpose(imarr,(2,0,1))
    imarrML = imarrML.reshape(3*height*width)
    imarrML = imarrML[np.newaxis,:]
    test_x = np.concatenate((test_x, imarrML),axis=0)

    val_y=np.array([0,1], dtype='int32')
    test_y=np.array([0,1], dtype='int32')

    dataset={'train':[[],[]], 'valid':[[],[]], 'test':[[],[]]}
    dataset['train']=[my_x, my_y]
    dataset['valid']=[val_x, val_y]
    dataset['test']=[test_x, test_y]
    f=gzip.open('my_data.pkl.gz', 'wb')
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
