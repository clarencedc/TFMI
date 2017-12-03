#coding=utf8
import os

import tensorflow as tf


# initialize variables/model parameters
# 初始化变量/模型参数
W = tf.Variable(tf.zeros([2, 1], name='weights'))
b = tf.Variable(0., name='bias')


# define the training loop parameters
def inference(X):
	# compute inference model over data X and return the result
	value = tf.matmul(X, W)+b
	# print 'b.shape', b.shape
	# print 'value.shape', value.shape
	return value

def loss(X, Y):
	# compute loss over training data X and expected outputs Y
	Y_predicted = inference(X)
	return tf.reduce_mean(tf.squared_difference(Y, Y_predicted))

def inputs():
	# read/generate input training data X and expected outputs Y
	weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57],
	              [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
	              [27, 24], [89, 31], [65, 52], [57, 23], [59, 60],
	              [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
	              [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
	blood_fat_content = [354, 190, 405, 263, 451,
	                     302, 288, 385, 402, 365,
	                     209, 290, 346, 254, 395,
	                     434, 220, 374, 308, 220,
	                     311, 181, 274, 303, 244]
	return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_loss):
	# train / adjust model parameters according to computed total loss
	learning_rate = 0.00001
	return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
	# evaluate the resulting trained model
	print sess.run(inference([[80., 25.]]))   # ~ 303
	print sess.run(inference([[65., 25.]]))   # ~ 256

# Create a saver.
saver = tf.train.Saver()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
	tf.initialize_all_variables().run()

	X, Y = inputs()

	total_loss = loss(X, Y)
	train_op = train(total_loss)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	initial_step = 0
	# verify if we don't have a checkpoint saved already
	print __file__
	print os.path.dirname(__file__)
	ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.dirname(__file__), 'checkpoint'))
	if ckpt and ckpt.model_checkpoint_path:
		# Restores from checkpoint
		saver.restore(sess, ckpt.model_checkpoint_path)
		initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

	# actual traning loop
	print 'initail_step:', initial_step
	training_steps = 3500000
	for step in range(initial_step, training_steps):
		_, loss = sess.run([train_op, total_loss])
		# for debugging and learning purposes, see how the loss gets decremented thru training steps
		if step % 10 == 0:
			print "loss:", loss

		if step % 1000 == 0:
			saver.save(sess, 'checkpoint/my-model', global_step=step)


	evaluate(sess, X, Y)

	saver.save(sess, 'checkpoint/my-model', global_step=training_steps)

	coord.request_stop()
	coord.join(threads)
	sess.close()