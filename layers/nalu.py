import tensorflow as tf


class NALU(object):
	def __init__(self, output_len,initializer=tf.contrib.layers.xavier_initializer()):
		assert output_len > 0
		self.output_len = output_len
		self.initializer = initializer
		self._gate_size = 1

	def __call__(self,input,scope="NALU"):
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			feature_size = input.get_shape().as_list()[-1]
			self._W = tf.get_variable("W_hat",shape=[feature_size,self.output_len],dtype=tf.float32,
										initializer=self.initializer,trainable=True)
			self._M = tf.get_variable("M_hat",shape=[feature_size,self.output_len],dtype=tf.float32,
										initializer=self.initializer,trainable=True)
			self._G = tf.get_variable("G_hat",shape=[feature_size,self._gate_size],dtype=tf.float32,
										initializer=self.initializer,trainable=True)

			self._const_W = tf.nn.tanh(self._W) * tf.nn.sigmoid(self._M)

			a = tf.matmul(input,self._const_W)
			m = tf.exp(tf.matmul(tf.log(tf.abs(input) + 0.00001),self._const_W))
			g = tf.nn.sigmoid(tf.matmul(input,self._G))

			return g * a + (1-g) * m

	def variables(self):
		return [self._W,self._M,self._G]

	def to_json(self,sess):
		curr_layer = {}
		Wval, Gval = sess.run([self._const_W,self._G])
		curr_layer['W'] = Wval.tolist()
		curr_layer['G'] = Gval.tolist()
		curr_layer['in_dim'] = Wval.shape[0]
		curr_layer['out_dim'] = Wval.shape[1]
		curr_layer['type'] = "NALU"
		return curr_layer
