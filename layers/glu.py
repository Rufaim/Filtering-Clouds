import tensorflow as tf
from .dense import Dense
from .utils import _activation_to_string

class GLUv2(object):
	def __init__(self,initializer=tf.contrib.layers.xavier_initializer()):
		self.initializer = initializer
	def __call__(self,input,scope="GLU"):
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			feature_size = input.get_shape().as_list()[-1]
			self._inner_dense = Dense(feature_size,tf.nn.sigmoid,initializer=self.initializer)
			return input * self._inner_dense(input)
	def to_json(self,sess):
		curr_layer = self._inner_dense.to_json(sess)
		curr_layer['type'] = "GLUv2"
		return curr_layer


class GLU(object):
	def __init__(self,activation=tf.identity):
		self.activation = activation
	def __call__(self,input,scope="GLU"):
		with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
			self._in_feature_size = input.get_shape().as_list()[-1]
			assert self._in_feature_size % 2 == 0, "input last dimention should be divisible by two" 
			self._out_feature_size = self._in_feature_size // 2
			return self.activation(input[:,:self._out_feature_size]) * tf.nn.sigmoid(input[:,self._out_feature_size:])
	def to_json(self,sess):
		curr_layer={}
		curr_layer['in_dim'] = self._in_feature_size
		curr_layer['out_dim'] = self._out_feature_size
		curr_layer['activation'] = _activation_to_string(self.activation)
		curr_layer['type'] = "GLU"
		return curr_layer