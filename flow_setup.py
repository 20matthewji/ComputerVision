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
