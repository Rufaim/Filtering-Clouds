import tensorflow as tf
from .layer import Layer
from .utils import _activation_to_string

class Dense(Layer):
	def __init__(self,num_units,activation=tf.identity,initializer = tf.contrib.layers.xavier_initializer()):
		self.num_units = num_units
		self.activation = activation
		self.initializer = initializer
	def __call__(self,input,scope="Dense"):
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			feature_size = input.get_shape().as_list()[-1]

			self._W = tf.get_variable("W",shape=[feature_size,self.num_units],dtype=tf.float32,
										initializer=self.initializer,trainable=True)
			self._b = tf.get_variable("b",shape=[self.num_units],dtype=tf.float32,
										initializer=self.initializer,trainable=True)

			return self.activation(tf.matmul(input,self._W)+self._b)

	def variables(self):
		return [self._W, self._b]

	def to_json(self,sess):
		curr_layer = {}
		Wval, bval = sess.run([self._W, self._b])
		curr_layer['W'] = Wval.tolist()
		curr_layer['b'] = bval.tolist()
		curr_layer['in_dim'] = Wval.shape[0]
		curr_layer['out_dim'] = Wval.shape[1]
		curr_layer['activation'] = _activation_to_string(self.activation)
		curr_layer['type'] = "DENSE"
		return curr_layer
