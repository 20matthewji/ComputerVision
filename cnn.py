#!/usr/bin/python

from PIL import Image
from PIL import ImageFilter
from random import randint

import numpy as np
import pylab
import mahotas as mh

dir='/Users/haimingjin/Pictures/Strt/'
n='Park_Ave'
## Direct image cropping, removing background
img = Image.open(dir+n+'.gray.png')
img -= mh.thresholding.otsu(img)

## 0-Centralize --always +ve or -ve results in constant signed gradient on W
## Subtract mean X[N x D] is input data
X = img - np.mean(img, axis=0)
# Batch Normalization
X /= np.std(X, axis=0)

Xvec=np.reshape(X, X.size, 'C')

#y=[0, 0, 0, 1, 1, 1, 2, 2, ...] label of the object class
def hinge_SVM_loss(x, y, W, R):
#  scores = np.dot(x,W)  # x^T x W 
  scores = np.dot(W,x)
  margins = np.maximum(0.0, scores - scores[y]+1)
  margins[y]=0
  loss_vec = np.sum(margins)
  data_loss = np.average(hinge_loss)
  reg_loss = R * np.sum(W*W)
  return data_loss+reg_loss


def eval_numeric_gradient(f,x):
## Evaluate function value at original point
  fx = f(x)
  grad = np.zeros(x.shape)
  h = 0.00001
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value+h
    fxh = f(x)
    x[ix] = old_value
    grad[ix]=(fxh-fx)/h
    it.iternext()
  return grad

## Gradient Descent
while 1:
  data_batch = sample_training_data(data, 256)
  weights_grad = evaluate_gradient(loss_func, data_batch, weights)
  weights += -step_size * weight_grad

def softmax_classifier():

def two_layer_net(X_train, model, y_train, 1e3):
   ## input pixel scores (2D)
    score1 = np.dot(Xvec, model['W1'])
    relu = score1
    for i in range (0,scroe1.shape[0]):
      for j in range (0,score1.shape[1]):
          relu[i][j] = max(relu[i][j], 0)

    svm2 = np.dot(relu, model['W2'])
    return (loss,grad)

D,H=10000,50
# W=0.01 * np.random.randn(D,H)
## Xavier initialization
# W= np.random.randn(D,H)/np.sqrt(D)

hidden_layer_sizes = [500]*10;
nonlinearities = ['tanh']*len(hidden_layer_sizes)
act={'relu':lambda x:np.maximum(0,x), 'tanh':lambda x:np.tanh(x)}
Hs = {}

def init_two_layer_model(input_size, hidden_size,output_size):
  #initialize a model
  model={}
  model['W1'] = 0.0001 * np.random.randn(input_size, hidden_size) ##layer init
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.0001 * np.random.randn(hidden_size, output_size) ##layer init
  model['b2'] = np.zeros(output_size)
  return model

trainer = ClassifierTrainer()
X_tiny = X_train[:20]
y_tiny = y_train[:20]
best_model, stats = trainer.tran(X_tiny, y_tiny, X_tiny, y_tiny,
                                 model, two_layr_net, num_epochs=200,
                                 reg=0.0, update='sgd', learning_rate_decay=1,
                                sample_batches=False, learning_rate=1e-3, verbose=True)

for i in xrange (len(hidden_layer_sizes)):
  X=D if i==0 else Hs[i-1]
  fan_in = X.shape[1]
  fan_out = hidden_layer_sizes[i]
  W = np.random.randn(fanin, fanout)*0.01 ##layer init

  H = np.dot(X,W)	# Matrix multiply
  H = act[nonlinearities[i]](H)	#nonlinearity
  Hs[i] =H

print 'input layer had mean %f and std %f' %(np.mean(D), np.std(D))
layer_means = [np.mean(H) for i,H in Hs.iteritems()]
layer_stds = [np.std(H) for i,H in Hs.iteritems()]
for i,H in Hs.iteritems():
   print 'hidden layer %d had mean %f and std %f' %(i+1,layer_means[i],layer_stds[i])



