import tensorflow as tf
from .layer import Layer
from .utils import _activation_to_string

class SimpleAttention(Layer):
	def __init__(self,attention_size,activation=tf.identity,initializer = tf.contrib.layers.xavier_initializer()):
		self.attention_size = attention_size
		self._activation = activation
		self.initializer = initializer

	def __call__ (self,input,scope="SimpleAttention"):
		self.feature_size = input.get_shape().as_list()[-1]
		self.time_size = input.get_shape().as_list()[-2]
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			self._w_omega = tf.get_variable("W_omega",shape=[self.feature_size,self.attention_size],dtype=tf.float32,
											initializer=self.initializer,trainable=True)
			self._b_omega = tf.get_variable("b_omega",shape=[self.attention_size],dtype=tf.float32,
											initializer=self.initializer,trainable=True)
			self._u_omega = tf.get_variable("u_omega",shape=[self.attention_size],dtype=tf.float32,
											initializer=self.initializer,trainable=True)
			# input BxTxN
			v = tf.nn.tanh(tf.tensordot(input, self._w_omega, axes=1) + self._b_omega) # BxTxA
			vu = tf.tensordot(v, self._u_omega, axes=1, name='vu') # BxT
			alphas = tf.nn.softmax(vu, name='alphas')	# BxT
			output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), 1) # BxN

			return self._activation(output)

	def to_json(self,sess):
		curr_layer = {}
		Wval, bval,uval = sess.run([self._w_omega, self._b_omega,self._u_omega])
		curr_layer['W'] = Wval.tolist()
		curr_layer['b'] = bval.tolist()
		curr_layer['u'] = uval.tolist()
		curr_layer['in_dim'] = self.feature_size
		curr_layer['out_dim'] = Wval.shape[1]
		curr_layer['time_dim'] = self.time_size
		curr_layer['activation'] = _activation_to_string(self._activation)
		curr_layer['type'] = "SIMPLE_ATTENTION"
		return curr_layer