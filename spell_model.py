from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell

class Model(object):
	def __init__(self, hidden_size = 128, num_layers =2, chars_size = 34, l2_reg_lambda=0.0):
		
			self.sentence_x = tf.placeholder(tf.float32, shape = [None,None,34])
			self.sentence_y = tf.placeholder(tf.float32, shape = [None,34])

			input_shape = tf.shape(self.sentence_x)
			l2_loss = tf.constant(0.0)

			with tf.name_scope("lstm"):
				f_cell = BasicLSTMCell(hidden_size)
				#b_cell = BasicLSTMCell(hidden_size)

				#f_cell = MultiRNNCell([f_cell] * num_layers)
				#b_cell = MultiRNNCell([b_cell] * num_layers)
				
				lstm_x,_ = tf.nn.dynamic_rnn(f_cell, inputs = self.sentence_x, dtype = tf.float32)

				W = tf.Variable(tf.random_normal(shape = [hidden_size,chars_size],stddev =0.1), name = "W")
				b = tf.Variable(tf.constant(0.1, shape = [chars_size]), name = "b")

				#out = tf.add(tf.matmul(tf.reshape(inp,[-1, hidden_size]),W),b)

				inp = tf.reduce_max(lstm_x, axis = 1)
				
				scores = tf.add(tf.matmul(tf.reshape(inp,[-1, hidden_size]),W),b)   # [batch_size,chars_size]

				self.predictions = tf.argmax(scores,1,"predictions")

			"""
			with tf.name_scope("highway_layer"):
				W_t = tf.Variable(tf.truncated_normal(size = [hidden_size,hidden_size],stddev =0.1), name = "W_t")
				b_t = tf.Variable(tf.constant(0.1, shape = [hidden_size]), name = "b_t")

				W = tf.Variable(tf.truncated_normal(size = [hidden_size,hidden_size],stddev =0.1), name = "W")
				b = tf.Variable(tf.constant(0.1, shape = [hidden_size]), name = "b")

				t = tf.sigmoid(tf.matmul(tf.reshape(lstm_x, [-1, hidden_size]), W_t) + b_t, name="transform_gate")
				h = tf.tanh(tf.matmul(tf.reshape(lstm_x, [-1, hidden_size]), W) + b, name="activation")
				c = tf.sub(1.0, t, name="carry_gate")

				highway_x = tf.add(tf.mul(h, t), tf.mul(x, c))				
			"""

			with tf.name_scope('loss'):
				losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.sentence_y,logits = scores)
				self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.sentence_y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


