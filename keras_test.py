import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.models import model_from_json
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import pickle
import sys
import heapq
import os
import sys

SEQUENCE_LENGTH = 40

# load json and create model
json_file = open('/home/ujjawal/spell/saved_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/ujjawal/spell/saved_models/model.h5")

#model.load_weigths('/home/ujjawal/spell/saved_models/keras_model.h5')
#history = pickle.load(open("/home/ujjawal/spell/saved_models/history.p", "rb"))

with open('char2ind.pkl', 'rb') as handle:
	char_indices = pickle.load(handle)

indices_char = dict([(v,k) for k,v in dict.items(char_indices)])

def prepare_input(text):
	
	if(len(text)>SEQUENCE_LENGTH):
		text = text[-40:]
	x = np.zeros((1, SEQUENCE_LENGTH, len(char_indices)))
	for t, char in enumerate(text):
		
		x[0, t, char_indices[char]] = 1
	return x

prepare_input("This is an example of input for our LSTM".lower())

def sample(preds, top_n=3):
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds)
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	
	return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
	original_text = text
	generated = text
	completion = ''
	while True:
		x = prepare_input(text)
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, top_n=1)[0]
		next_char = indices_char[next_index]
		text = text[1:] + next_char
		completion += next_char
		
		if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
			return completion

def predict_completions(text, n=3):
	x = prepare_input(text)
	preds = model.predict(x, verbose=0)[0]
	next_indices = sample(preds, n)
	return [ text.split()[-1] + indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

quotes = [
	"It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
	"That which does not kill us makes us stronger.",
	"I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
	"And those who were seen dancing were thought to be insane by those who could not hear the music.",
	"It is hard enough to remember my opinions, without also remembering my reasons for them!"
	,"This is not fun at all.",
	"I did not get a chance to visit my dream place with my mother."
]

for q in quotes:
	for i in range(0,len(q)-SEQUENCE_LENGTH-1,2):
		seq = q[i:i+SEQUENCE_LENGTH].lower()
		print(seq)
		print(predict_completions(seq, 5))
	