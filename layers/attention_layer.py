import tensorflow as tf
from .self_attention import linear


class AttentionLayer(object):
	def __init__ (self,self_attention, num_units_linear, nonlinearity=tf.nn.selu):
		self.self_attention = self_attention
		self.num_units_linear = num_units_linear
		self.nonlinearity = nonlinearity

	def __call__(self,input,scope = "AttentionLayer"):
		max_len,features_size = tuple(input.get_shape().as_list()[-2:])

		with tf.variable_scope(scope):
			out = self.self_attention(input,input,input)
			out = linear(input,self.self_attention.attention_dim) + out
			out = tf.contrib.layers.layer_norm(out)

			b = tf.get_variable("b",[self.num_units_linear],dtype=tf.float32)

			out = self.nonlinearity(linear(out,self.num_units_linear,scope="linear_out") + b)
			return out
