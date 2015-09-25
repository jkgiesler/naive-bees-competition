import cPickle as pickle

import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet

import theano
import theano.tensor as T

from PIL import Image
import glob
import cPickle as pickle
import numpy

theano.allow_input_downcast=True


#Constants
num_epochs = 3
batch_size = 500 #make sure your total images are divisible by your batch size
learning_rate = 0.01
momentum = 0.9
pic_width = pic_height = 200
output_dim = 2

convNet = NeuralNet(
	layers=[
	('input',layers.InputLayer),
	('conv1',layers.Conv2DLayer),
	('pool1',layers.MaxPool2DLayer),
	('conv2',layers.Conv2DLayer),
	('pool2',layers.MaxPool2DLayer),
	('hidden1',layers.DenseLayer),
	('dropout1',layers.DropoutLayer),
	('output',layers.DenseLayer),
	],

	input_shape = (None,3,pic_width,pic_height),
	conv1_num_filters =32,
	conv1_filter_size=(5,5),

	pool1_pool_size=(2,2),

	conv2_num_filters=32,
	conv2_filter_size=(5,5),

	pool2_pool_size=(2,2),

	hidden1_num_units=256,

	dropout1_p=.5,

	output_num_units = output_dim,
	output_nonlinearity=lasagne.nonlinearities.softmax,

	update_learning_rate = learning_rate,
	update_momentum = momentum,
	regression = False,

	max_epochs = num_epochs,
	verbose=1,
)

def generate_data():
	'''
	Pictures quickly eat up RAM so we are only loading in a set amount at a time.
	In theory we could directly call the picture_generator function here. This
	will probably be done in the future as we then wouldn't have to rely on 
	a static directory of photos. We end yielding a numpy object.
	'''
	data = []
	labels = []
	count = 0
	#replace with your path to generated images
	images = glob.glob('../bee_project/*.jpg')
	for f_name in images:
		if count >= batch_size:
			X_train = numpy.array(data,dtype='float32')
			y_train = numpy.array(labels,dtype='int32')
			data = []
			labels = []
			count = 0
			yield (X_train,y_train)

		im = Image.open(f_name,mode='r')
		dat = numpy.asarray(im).astype('float32') / 255
		im.close()
		dat = numpy.rollaxis(dat,2)
		data.append(dat)
		
		#hacky way of getting classification from filename
		jpg_idx = f_name.find('.jpg')
		labels.append(int(f_name[jpg_idx-1:jpg_idx]))

		count+=1

for X_epoch,y_epoch in generate_data():
	#fit the batchsize to our model
	convNet.partial_fit(X_epoch,y_epoch)

with open('convNet.pickle','wb') as f:
	#save our model to a file
	pickle.dump(convNet,f,-1)