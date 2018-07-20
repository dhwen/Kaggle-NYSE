import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import os
import shutil
import numpy as np
from dataloader import DataLoader
from model import Model

export_path = 'saved_model/'
ckpt_path = 'ckpt_model/'
LOGDIR='logs/'
batch_size = 1

if os.path.isdir(export_path):
	shutil.rmtree(export_path)
if not os.path.isdir(ckpt_path):
	os.makedirs(ckpt_path)
if os.path.isdir(LOGDIR):
	shutil.rmtree(LOGDIR)
	
builder = saved_model_builder.SavedModelBuilder(export_path)

model = Model()
loader = DataLoader()
[train_input, train_labels] = loader.load(file_name='data/aapl.us.csv')

print(len(train_input[0]))
print(len(train_labels[0]))

#//Model Run//

with tf.Session(graph=model.graph) as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	
	#/ Restore ckpt file if exists
	#if os.path.isfile(ckpt_path + 'SmallNet.ckpt.meta'):
	#	saver.restore(sess, tf.train.latest_checkpoint(ckpt_path)) 
	#	print "restored"

	#/Train Network
	i=0
	num_epochs = np.power(10, 5)
	for _ in range(num_epochs):
		i=i+1
		[out, loss] = sess.run([model.opt, model.loss], feed_dict={model.input: train_input, model.label: train_labels})
		print('Epoch %d, training accuracy is %g' % (i, loss))

	
	#/Save ckpt file
	saver.save(sess,ckpt_path + "SmallNet.ckpt")