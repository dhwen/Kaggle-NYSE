import tensorflow as tf

class Model:

	def __init__(self, dropout_keep_prob=1):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.build_layers(dropout_keep_prob)
			self.build_backprop()

	def build_layers(self, dropout_keep_prob):
		self.input = tf.placeholder(dtype="float32", shape=(None,4), name="Input")
		self.fc1 = tf.layers.dense(self.input, 6, name="FC1")
		self.relu1 = tf.nn.relu(self.fc1, name="Relu1")
		self.fc2 = tf.layers.dense(self.relu1, 4, name="FC2")
		self.relu2 = tf.nn.relu(self.fc2, name="Relu2")
		self.fc3 = tf.layers.dense(self.relu2, 1, name="FC3")
		self.output = tf.nn.relu(self.fc3, name="Output")

	def build_backprop(self):
		self.label = tf.placeholder(dtype="float32", shape=(None, 1), name="Label")
		self.loss = tf.losses.mean_squared_error(self.label, self.output)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
		self.opt = self.optimizer.minimize(self.loss)