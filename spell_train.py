from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import datetime
from spell_model import Model
from data_stream import *



# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("val_percentage", 0.1, "Validation Percentage(default: 0.1)")
tf.flags.DEFINE_integer("starter_learning_rate",0.0088, "learning rate")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{} = {}".format(attr.upper(), value))
print("")

with tf.Graph().as_default():

	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True

	#sess = tf.Session(config = config)

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	sess = tf.Session(config=config)


	with sess.as_default():
		model = Model()

		global_step = tf.Variable(0, name="global_step", trainable=False)

		optimizer  = tf.train.AdamOptimizer(FLAGS.starter_learning_rate)
		grads_and_vars = optimizer.compute_gradients(model.loss)

		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary =  tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)

		#Set log output directory
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))

		# create a summary for our cost and accuracy
		loss_summary = tf.summary.scalar("loss", model.loss)
		acc_summary = tf.summary.scalar("accuracy", model.accuracy)

		
		train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer =  tf.summary.FileWriter(train_summary_dir, sess.graph)

		dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir, sess.graph)

		#summary_op = tf.merge_all_summaries()
		#summary_writer = tf.train.SummaryWriter('/home/', graph_def=sess.graph)

		def train_step(x_batch,y_batch):
			feed_dict  = {
				model.sentence_x : x_batch,
				model.sentence_y : y_batch
			}

			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, model.loss, model.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {},loss {:g}, acc {:g}".format(time_str, step,loss, accuracy))
			train_summary_writer.add_summary(summaries, step)


		def val_step(writer=None):
			v = get_batches(0.01)
			#length = len(rp[5200:])
			acc = []
			losses =[]
 #           print("Number of batches in dev set is " + str(length))
			while(True):
				try:
					x_batch_dev, y_batch_dev = next(v)

					feed_dict = {
					  model.sentence_x: x_batch_dev,
					  model.sentence_y: y_batch_dev
					}
					step,summaries, loss, accuracy = sess.run(
						[global_step, dev_summary_op, model.loss, model.accuracy],
						feed_dict)
					acc.append(accuracy)
					losses.append(loss)
					time_str = datetime.datetime.now().isoformat()
					#print(" in dev >>" +
					#	   " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
					if writer:
						writer.add_summary(summaries, step)
				except StopIteration:
					break

			print('##############################################')
			print(datetime.datetime.now().isoformat())
			print("\nMean accuracy=" + str(sum(acc)/len(acc)))
			print("Mean loss=" + str(sum(losses)/len(losses)))

		num_epoch = FLAGS.num_epochs

		g = get_batches(0.3)
		i=1
		print("Epoch >> "+ str(i))
		for epoch in range(num_epoch*10000000):
			try:
				x_batch ,y_batch = next(g)
				#print(x_batch.shape)
				train_step(x_batch, y_batch)
			except StopIteration:	
				saver.save(sess, os.path.join(os.path.abspath(os.path.join(os.path.curdir, "saved_models")),"spell_model"), global_step = i)
				i+=1
				g = get_batches(0.3)
				val_step(writer = dev_summary_writer)
				print('Epoch >> '+ str(i))
				pass