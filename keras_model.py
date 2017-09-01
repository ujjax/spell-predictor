import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
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
import json

text = open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","corpus.txt")))).read().lower()
text += open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","english-brown.txt")))).read().lower()
text += open(os.path.join(os.path.abspath(os.path.join(os.path.curdir, "data","word_list.txt")))).read().lower()

text = text[:int(0.1*len(text))]

print('corpus length:', len(text))

chars_ = [ u'$', u'%', u'&', u'*', u'+', u'-', u'/', u'0', u'1', u'2', u'3', u'4',u'5', u'6', u'7', u'8', u'9', u':', u';', u'=', u'[', u']', u'_', u'\xc6', u'\xe4', u'\xe6', u'\xe9', u'\xeb',u'\u2014']

text = text.replace("\n", " ").decode('utf-8')
for c_ in chars_:
	text = text.replace(c_, "")


chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

with open('char2ind.pkl', 'wb') as handle:
		pickle.dump(char_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

SEQUENCE_LENGTH = 40
step = 5
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])


X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(None, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=4, shuffle=True).history

#model.save('/home/ujjawal/spell/saved_models/keras_model.h5')
#pickle.dump(history, open("/home/ujjawal/spell/saved_models/history.p", "wb"))

# serialize model to JSON
model_json = model.to_json()
with open("/home/ujjawal/spell/saved_models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/ujjawal/spell/saved_models/model.h5")
print("Saved model to disk")
