import tensorflow as tf

class Model:

	def __init__(self, dropout_keep_prob=1):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.build_layers(dropout_keep_prob)
			self.build_backprop()

	def build_layers(self, dropout_keep_prob):
		self.input = tf.placeholder(dtype="float32", shape=(None,4), name="Input")
		self.fc1 = tf.layers.dense(self.input, 7, name="FC1")
		self.bn1 = tf.layers.batch_normalization(self.fc1, name="BN1")
		self.relu1 = tf.nn.relu(self.bn1, name="Relu1")
		self.dropout1 = tf.layers.dropout(self.relu1, name="DropOut1")
		self.fc2 = tf.layers.dense(self.dropout1, 5, name="FC2")
		self.bn2 = tf.layers.batch_normalization(self.fc2, name="BN2")
		self.relu2 = tf.nn.relu(self.bn2, name="Relu2")
		self.dropout2 = tf.layers.dropout(self.relu2, name="DropOut2")
		self.fc3 = tf.layers.dense(self.dropout2, 4, name="FC3")
		self.bn3 = tf.layers.batch_normalization(self.fc3, name="BN3")
		self.relu3 = tf.nn.relu(self.bn3, name="Relu3")
		self.dropout3 = tf.layers.dropout(self.relu1, name="DropOut3")
		self.fc4 = tf.layers.dense(self.dropout3, 2, name="FC4")
		self.output = tf.nn.softmax(self.fc4, name="Output")

	def build_backprop(self):
		self.label = tf.placeholder(dtype="float32", shape=(None, 2), name="Label")
		self.loss = tf.losses.softmax_cross_entropy(self.label, self.fc4)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
		self.opt = self.optimizer.minimize(self.loss)