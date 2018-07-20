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
[train_input, train_labels] = loader.load(file_name='Stocks/aapl.us.txt',row_start=2,row_end=6000)
[test_input, test_labels] = loader.load(file_name='Stocks/aapl.us.txt',row_start=7501,row_end=8000)

for label in train_labels:
	print(label)

#//Model Run//

with tf.Session(graph=model.graph) as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	
	#/ Restore ckpt file if exists
	#if os.path.isfile(ckpt_path + 'SmallNet.ckpt.meta'):
	#	saver.restore(sess, tf.train.latest_checkpoint(ckpt_path)) 
	#	print "restored"

	#/Train Network
	i = 0
	num_epochs = np.power(10, 4)
	for _ in range(num_epochs):
		i=i+1
		[opt, output, loss] = sess.run([model.opt, model.output, model.loss], feed_dict={model.input: train_input, model.label: train_labels})
		print('Epoch %d, training loss is %g' % (i, loss))

		[output, loss] = sess.run([model.output, model.loss], feed_dict={model.input: test_input, model.label: test_labels})
		print('Epoch %d, test loss is %g' % (i, loss))

	np.savetxt('test_labels.txt', np.array(test_labels, dtype=np.float))
	np.savetxt('test_output.txt', output)

	#/Save ckpt file
	saver.save(sess,ckpt_path + "SmallNet.ckpt")