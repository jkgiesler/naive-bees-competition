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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

batch_size = 500

def check_img_data():
	'''makes sure all of the images are readable/writeable I don't know if this 
	fixed a weird bug I was having but after I ran it everything worked
	'''
	images = glob.glob('../bee_project/bee_images/test/*.jpg')
	for i in images:
		im = Image.open(i,mode='r')
		im.save(i)

def generate_train_data():
	data = []
	labels = []
	count = 0
	#location of generated images
	images = glob.glob('../BEEEES/*.jpg')[:30000]
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

	yield (X_train,y_train)

def generate_test_data():
	data = []
	count = 0
	#location of test images
	images = glob.glob('../bee_images/test/*.jpg')
	print(len(images))
	for f_name in images:
		if count >= batch_size:
			print 'yield'
			X_test = numpy.array(data,dtype='float32')
			data = []
			count = 0
			yield X_test

		im = Image.open(f_name,mode='r')
		dat = numpy.asarray(im).astype('float32') / 255
		im.close()
		dat = numpy.rollaxis(dat,2)
		data.append(dat)
		count+=1
	yield X_test




with open('convNet.pickle','rb') as f:
	model = pickle.load(f)

###really crummy practice of checking model efficiency on test data
###TODO(jkg): implement validation set holdout (could I just generate one?)
for X_train,truth in generate_train_data():
	guess = model.predict(X_train)
	print classification_report(guess,truth,['0','1'])
	print confusion_matrix(guess,truth)

'''
###for real test data
###TODO(jkg): append guesses and generate nice outfile
for X_test in generate_test_data():
	guess = model.predict(X_test)
	print guess
'''