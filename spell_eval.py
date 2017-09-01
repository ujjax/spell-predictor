from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import sys

from spell_model import Model
from data_stream import *


with open('char2ind.pkl','r') as f:
   c2i = pickle.load(f)


i2c = dict([v,k] for k,v in dict.items(c2i))

def fetch_labels(lis):
	
	l = [c2i[w] for w in lis]
	return l

def fetch_x(sentence):
	sentence = fetch_labels(sentence)
	if(len(sentence)>40):
		sentence = sentence[-40:]
		X = np.zeros((1,40, 34), dtype=np.int32)
		for i,t in enumerate(sentence):
			X[0,i,t] =1
		return X
	else:
		X = np.zeros((1,len(sentence), 34), dtype=np.int32)
		for i,t in enumerate(sentence):
			X[0,i,t] =1
		return X

def test_step(batch_x):
	feed_dict = {model.sentence_x: batch_x}
	pred = sess.run(model.predictions,feed_dict)
	
	return pred

sentence = sys.argv[1].lower()
#print(c2i,sentence)
X = fetch_x(sentence)


with tf.Session() as sess:
	model = Model()
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,'/home/ujjawal/spell/saved_models/spell_model-5')
	print(test_step(X))
	
