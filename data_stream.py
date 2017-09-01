
# coding: utf-8

from __future__ import print_function

import csv
import os
import numpy as np
import nltk
import pickle
import random

extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e']
#sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
tokenizer._params.abbrev_types.update(extra_abbreviations)

def get_data_(percent = None):
	text = open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","corpus.txt")))).read().lower()
	text += open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","english-brown.txt")))).read().lower()
	
	chars_ = [ u'$', u'%', u'&', u'*', u'+', u'-', u'/', u'0', u'1', u'2', u'3', u'4',u'5', u'6', u'7', u'8', u'9', u':', u';', u'=', u'[', u']', u'_', u'\xc6', u'\xe4', u'\xe6', u'\xe9', u'\xeb',u'\u2014']
	
	text = text.replace("\n", " ").decode('utf-8')
	for c_ in chars_:
		text = text.replace(c_, "")

	sentences_tokenized = [" ".join(se.split()) for se in list(set(tokenizer.tokenize(text)))]
	
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	
	with open('char2ind.pkl', 'wb') as handle:
		pickle.dump(char_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	sentences = []
	next_chars = []
	for sent in sentences_tokenized:
			i =20
			try:
				if(len(sent[:i].split()[-1])>=3):
					sentences.append(sent[:i+1])
			except IndexError:
				break

	
	random.seed(123)
	random.shuffle(sentences)
	print(len(sentences),"£££££££££££££££££££££££££££££££££££££££££££")
	lev = int(percent*len(sentences))
	sentences = sentences[:lev]
	#sentences = sentences[:10000]
	print(len(sentences),"#########################################")
	sentences.sort(key = lambda s: len(s))

	return sentences , char_indices, chars

def get_batches_(percent=None):
		sentences, char_indices, chars = get_data(percent)
		#for length in range(10,40):
		length = 20
		sent = [word for word in sentences if len(word) == length +1]
		len_batch = 128
		X = np.zeros((len(sent),length, len(chars)), dtype=np.int32)
		y = np.zeros((len(sent), len(chars)), dtype=np.int32)

		for i, sentence in enumerate(sent):
			for t, char in enumerate(sentence[:-1]):
				X[i, t, char_indices[char]] = 1
			y[i, char_indices[sentence[-1]]] = 1

		for j in range(len(sent)/130):
			yield [X[j*len_batch : (j+1)*len_batch], y[j*len_batch : (j+1)*len_batch]]

SEQUENCE_LENGTH = 20

with open('/home/ujjawal/spell/char2ind.pkl','r') as f:
		c2i = pickle.load(f)


def get_data():
	text = open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","corpus.txt")))).read().lower()
	#text += open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","english-brown.txt")))).read().lower()
	
	chars_ = [ u'$', u'%', u'&', u'*', u'+', u'-', u'/', u'0', u'1', u'2', u'3', u'4',u'5', u'6', u'7', u'8', u'9', u':', u';', u'=', u'[', u']', u'_', u'\xc6', u'\xe4', u'\xe6', u'\xe9', u'\xeb',u'\u2014']
	
	text = text.replace("\n", " ").decode('utf-8')
	for c_ in chars_:
		text = text.replace(c_, "")
	
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	
	with open('char2ind.pkl', 'wb') as handle:
		pickle.dump(char_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	step = 5
	sentences = []
	
	for i in range(0, len(text) - SEQUENCE_LENGTH, step):
			sentences.append(text[i: i + SEQUENCE_LENGTH+1])
	print('num training examples: {}'.format(len(sentences)))

	import random

	random.seed(123)
	random.shuffle(sentences)

	sentences = sentences[:int(0.001*len(sentences))]

	return sentences, char_indices



def get_batches(num=None):
	sentences = get_data()
	i=0
	for i in range(len(sentences)//128):

		X = np.zeros((128, SEQUENCE_LENGTH, len(c2i)), dtype=np.int32)
		y = np.zeros((128, len(c2i)), dtype=np.int32)
		
		for sentences_ in sentences[i*128:(i+1)*128]:
			for sentence in sentences_:	
				for t, char in enumerate(sentence[:-1]):
					X[i, t, c2i[char]] = 1
				y[i, c2i[sentence[-1]]] = 1
			yield [X,y]
		
