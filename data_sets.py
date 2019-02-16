
import numpy as np
import cPickle, gzip

from PIL import Image
from PIL import ImageFilter
from random import randint
from matplotlib import pyplot as plt

test_set = np.array([])
valid_set = np.array([])
train_set = np.array([])

groups=3
dir='/Users/Matthew/Documents/workspace/SRC/Strt/'
for name in ('Asbury_Ave', 'Dumas_Dr', 'Mendocino_Ln', 'Monterey_Ct', 'Pacific_Ave', 'Park_Ave', 'Regan_Ln', 'Roosevelt_Ave'):
	im = Image.open(dir+name+".gray.png")
	imarr = np.asarray(im)/256.
	imarr = np.append(imarr.reshape(100*100),0)
	imarr = imarr[np.newaxis,:]
	if (train_set.size==0):
		train_set = imarr
	else:
		train_set = np.concatenate((train_set, imarr),axis=0)

im = Image.open(dir+"Terra_Bella_Dr.gray.png")
#im = Image.open(dir+"Dumas_Dr.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),0)
imarr = imarr[np.newaxis,:]
if (valid_set.size==0):
	valid_set = imarr
else:
	valid_set = np.concatenate((valid_set, imarr),axis=0)

im = Image.open(dir+"Terrace_Dr.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),0)
imarr = imarr[np.newaxis,:]
if (test_set.size==0):
        test_set = imarr
else:
        test_set = np.concatenate((test_set, imarr),axis=0)

dir='/Users/Matthew/Documents/workspace/SRC/Obs/'
for name in ('Newport_tree', 'Castleton_myrtle', 'Castleton_palm', 'Dumas_fyard', 'Fairview_post', 'Harker_Obs', 'Newport_tree', 'Palm_post'):
	im = Image.open(dir+name+".gray.png")
	imarr = np.asarray(im)/256.
	imarr = np.append(imarr.reshape(100*100),1)
	imarr = imarr[np.newaxis,:]
	if (train_set.size==0):
		train_set = imarr
	else:
		train_set = np.concatenate((train_set, imarr),axis=0)

im = Image.open(dir+"Wacker_obs.gray.png")
# im = Image.open(dir+"Castleton_palm.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),1)
imarr = imarr[np.newaxis,:]
if (valid_set.size==0):
        valid_set = imarr
else:
        valid_set = np.concatenate((valid_set, imarr),axis=0)

im = Image.open(dir+"Washington_post.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),1)
imarr = imarr[np.newaxis,:]
if (test_set.size==0):
        test_set = imarr
else:
        test_set = np.concatenate((test_set, imarr),axis=0)


dir='/Users/Matthew/Documents/workspace/SRC/House/'
for name in ('28thAve', 'BellemeadeStrt', 'KingsgateDr', 'ChopinDr', 'Dumas', 'GardenGateDr', 'GoldenGateDr', 'BerkeleyWay'):
	im = Image.open(dir+name+".gray.png")
	imarr = np.asarray(im)/256.
	imarr = np.append(imarr.reshape(100*100),2)
	imarr = imarr[np.newaxis,:]
	if (train_set.size==0):
		train_set = imarr
	else:
		train_set = np.concatenate((train_set, imarr),axis=0)

im = Image.open(dir+"RainbowDr.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),2)
imarr = imarr[np.newaxis,:]
if (valid_set.size==0):
        valid_set = imarr
else:
        valid_set = np.concatenate((valid_set, imarr),axis=0)

im = Image.open(dir+"ChopinDr.gray.png")
imarr = np.asarray(im)/256.
imarr = np.append(imarr.reshape(100*100),2)
imarr = imarr[np.newaxis,:]
if (test_set.size==0):
        test_set = imarr
else:
        test_set = np.concatenate((test_set, imarr),axis=0)

np.random.shuffle(train_set[:groups*8])
x = np.float32(train_set[:groups*8,:10000].reshape((groups*8,10000)))
train_lab = np.int32(train_set[:groups*8,10000:].reshape(groups*8))
train_set = x

np.random.shuffle(valid_set[:groups])
x = np.float32(valid_set[:groups, :10000].reshape((groups,10000)))
valid_lab = np.int32(valid_set[:groups,10000:].reshape(groups))
valid_set = x

np.random.shuffle(test_set[:groups])
x = np.float32(test_set[:groups,:10000].reshape((groups,10000)))
test_lab = np.int32(test_set[:groups,10000:].reshape(groups))
test_set = x

# valid_lab = np.array([0,1,2], dtype='int32')
# test_lab = np.array([0,1,2], dtype='int32')

dataFile = 'nnet_SOHgray.pkl.gz'
dataset={'train':[[],[]], 'valid':[[],[]], 'test':[[],[]]}
dataset['train']=[train_set,train_lab]
dataset['valid']=[valid_set,valid_lab]
dataset['test']=[test_set,test_lab]
f = gzip.open(dataFile,'wb')
cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
