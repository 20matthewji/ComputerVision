# coding: utf-8
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

##  dir='/Users/haimingjin/Pictures/Strt/'
##  name='Regan_Ln'
##  im=Image.open(dir+name+'.png')
##  imshow(im)
##  cropim=im.crop((0,0,700,700))
##  imshow(cropim)
##  cropim.thumbnail((100,100),Image.ANTIALIAS)
##  imshow(cropim)
##  cropim.save(dir+name+".thumb.png","PNG")
##  grayim=cropim.convert("L")
##  grayim.save(dir+name+".gray.png","PNG")
##  imshow(grayim)
##  pylab.imshow(grayim)
##  im=Image.open('../Pictures/Strt/Roosevelt_Ave.gray.png')
##  im=imshow(im)
##  grayim.save(dir+n+".gray.png", "PNG")
##  grayim.save(dir+name+".gray.png", "PNG")
##  edgeim=cropim.filter(ImageFilter.FIND_EDGES)
##  edgeim.save(dir+name+".edge.png", "PNG")
##  imshow(cropim)
##  cropim.save(dir+name+".thumb.png", "PNG")
##  dir='/Users/haimingjin/Pictures/Obs/'
##  name='Harker_Obs'
##  im=Image.open(dir+name+".png")
##  imshow(im)
##  cropim=im.crop((100,100,800,800))
##  imshow(cropim)
##  cropim.thumbnail((100,100),Image.ANTIALIAS)
##  imshow(cropim)
##  cropim.save(dir+name+".thumb.png", "PNG")
##  grayim=cropim.convert('L')
##  grayim.save(dir+name+".gray.png","PNG")
##  edgeim=cropim.filter(ImageFilter.FIND_EDGES)
##  edgeim.save(dir+name+".edge.png","PNG")
##  imshow(edgeim)
##  imshow(grayim)
##  dir='/Users/haimingjin/Pictures/Strt/'
##  name='Dumas_Dr'
##  im=Image.open(dir+name+'.png')
##  imshow(im)
##  cropim=im.crop((100,100,650,650))
##  imshow(cropim)
##  cropim.save(dir+name+".thumb.png","PNG")
##  grayim=cropim.thumbnail((100,100),Image.ANTIALIAS)
##  imshow(grayim)
##  cropim.save(dir+name+".thumb.png", "PNG")
##  grayim=cropim.convert("L")
##  imshow(grayim)
##  edgeim=cropim.filter(ImageFilter.FIND_EDGES)
##  grayim.save(dir+name+".gray.png", "PNG")
##  edgeim.save(dir+name+".edge.png", "PNG")
##  imshow(edgeim)
##  dir='/Users/haimingjin/Pictures/Obs/'
##  name='Dumas_fyard'
##  im=Image.open(dir+name+'.png')
##  imshow(im)
##  cropim=im.crop((200,200,1000,1000))
##  imshow(cropim)
##  cropim.thumbnail((100,100), Image.ANTIALIAS)
##  imshow(cropim)
##  cropim.save(dir+name+".thumb.png","PNG")
##  grayim=cropim.convert("L")
##  imshow(grayim)
##  grayim.save(dir+name+".gray.png","PNG")
##  edgeim=cropim.filter(ImageFilter.FIND_EDGES)
##  edgeim.save(dir+name+".edge.png", "PNG")
##  imshow(edgeim)
##  val_y=np.array([0,1], dtype=int'32')
##  val_y=np.array([0,1], dtype='int32')
##  val_y
##  val_y.shape
##  dir='/Users/haimingjin/Pictures/Strt/'
##  name="Regan_Ln"
##  im=Image.open(dir+name+".thumb.png")
##  imarr = np.asarray(im)/256.
##  height,width=imarr.shape[:2]
##  imarrML = np.transpose(imarr,(2,0,1))
##  imarrML = imarrML.reshape(3*height*width)
##  val_x = imarrML[np.newaxis,:]
##  name="Dumas_Dr"
##  im=Image.open(dir+name+".thumb.png")
##  imarr = np.asarray(im)/256.
##  height,width=imarr.shape[:2]
##  imarrML = np.transpose(imarr,(2,0,1))
##  imarrML = imarrML.reshape(3*height*width)
##  test_x = imarrML[np.newaxis,:]
##  dir='/Users/haimingjin/Pictures/Obs/'
##  name="Harker_Obs"
##  im=Image.open(dir+name+".thumb.png")
##  imarr = np.asarray(im)/256.
##  height,width=imarr.shape[:2]
##  imarrML = np.transpose(imarr,(2,0,1))
##  imarrML = imarrML.reshape(3*height*width)
##  imarrML = imarrML[np.newaxis,:]
##  val_x = np.concatenate((val_x, imarrML),axis=0)
##  name="Dumas_fyard"
##  im=Image.open(dir+name+".thumb.png")
##  imarr = np.asarray(im)/256.
##  height,width=imarr.shape[:2]
##  imarrML = np.transpose(imarr,(2,0,1))
##  imarrML = imarrML.reshape(3*height*width)
##  imarrML = imarrML[np.newaxis,:]
##  test_x = np.concatenate((test_x, imarrML),axis=0)
##  val_y=np.array([0,1], dtype='int32')
##  test_y=np.array([0,1], dtype='int32')
##  val_x.shape
##  val_y.shape
##  test_x.shape, test_y.shape
##  my_x = np.array([]);
##  dir='/Users/haimingjin/Pictures/Strt/'
##  for name in ('Asbury_Ave', 'Mendocino_Ln', 'Monterey_Ct', 'Pacific_Ave', 'Park_Ave', 'Roosevelt_Ave', 'Terra_Bella_Dr', 'Terrace_Dr'):
##      im=Image.open(dir+name+".thumb.png")
##      imarr = np.asarray(im)/256.
##      height,width=imarr.shape[:2]
##      imarrML = np.transpose(imarr,(2,0,1))   # same as imarr.swapaxes(0,-1) then (1, -1)
##      imarrML = imarrML.reshape(3*height*width)
##      imarrML = imarrML[np.newaxis,:]
##      if (my_x.size==0):
##         my_x = imarrML
##         # my_y = label
##      else:
##         my_x = np.concatenate((my_x, imarrML),axis=0)
##  my_y = np.zeros(len(my_x),dtype='int32')
##  dir='/Users/haimingjin/Pictures/Obs/'
##  label = np.ones([1], dtype='int32')
##  for name in ('Newport_tree', 'Castleton_myrtle', 'Castleton_palm', 'Palm_post', 'Wacker_obs', 'Fairview_post', 'Washington_post', 'Bubb_Redwood'):
##      im=Image.open(dir+name+".thumb.png")
##      imarr = np.asarray(im)/256.
##      height,width=imarr.shape[:2]
##      imarrML = np.transpose(imarr,(2,0,1))   # same as imarr.swapaxes(0,-1) then (1, -1)
##      imarrML = imarrML.reshape(3*height*width)
##      imarrML = imarrML[np.newaxis,:]
##      if (my_x.size==0):
##         my_x = imarrML
##         # my_y = label
##      else:
##         my_x = np.concatenate((my_x, imarrML),axis=0)
##  my_y = np.append(my_y, np.ones(len(my_x)-len(my_y), dtype='int32'))
##  my_x.shape, my_y.shape
##  dataset={'train':[[],[]], 'valid':[[],[]], 'test':[[],[]]}
##  dataset['train']=[my_x, my_y]
##  dataset['valid']=[val_x, val_y]
##  dataset['test']=[test_x, test_y]
##  dataset
##  f=gzip.open('my_data.pkl.gz', 'wb')
##  cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
##  f.close()
f=gzip.open('my_data.pkl.gz','rb')
datasets = cPickle.load(f)
train_set_x, train_set_y = datasets['train']
valid_set_x, valid_set_y = datasets['valid']
test_set_x, test_set_y = datasets['test']
train_set_x.shape, train_set_y.shape
valid_set_x.shape, valid_set_y.shape
test_set_x.shape, test_set_y.shape
n_train_batches=2
batch_size=8
valid_set_x.get_value(borrow=True).shape[0]
valid_set_x.shape[0]
n_train_batches=train_set_x.shape[0]
train_set_x.shape[1]
n_valid_batches=valid_set_x.shape[0]
n_train_batches
n_train_batches//=batch_size
n_train_batches
n_valid_batches//=batch_size
n_valid_batches
n_test_batches=n_valid_batches
index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
rng = numpy.random.RandomState(23455); nkerns=[20,50]; learning_rate=0.1
layer0_input = x.reshape((batch_size, 3, 100, 100))
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 3, 100, 100),
    filter_shape=(nkerns[0], 3, 21, 21),
    poolsize=(2, 2)
)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 80, 80),
    filter_shape=(nkerns[1], nkerns[0], 21, 21),
    poolsize=(2, 2)
)
layer2_input = layer1.output.flatten(2)
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 30 * 30,
    n_out=750,
    activation=T.tanh
)
layer3 = LogisticRegression(input=layer2.output, n_in=750, n_out=10)
cost = layer3.negative_log_likelihood(y)
sh_test_x=theano.shared(test_set_x)
sh_test_y=theano.shared(test_set_y)
sh_test_x.type
sh_test_x.get_value(borrow=True).shape
sh_valid_x.get_value(borrow=True).shape
test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: sh_test_x[index * batch_size: (index + 1) * batch_size],
        y: sh_test_y[index * batch_size: (index + 1) * batch_size]
    }
)
sh_valid_x=theano.shared(valid_set_x)
sh_valid_y=theano.shared(valid_set_y)
validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: sh_valid_x[index * batch_size: (index + 1) * batch_size],
        y: sh_valid_y[index * batch_size: (index + 1) * batch_size]
    }
)
params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)
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
        x: shared_x[index * batch_size: (index + 1) * batch_size],
        y: shared_y[index * batch_size: (index + 1) * batch_size]
    }
)
