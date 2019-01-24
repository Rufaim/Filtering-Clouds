import tensorflow as tf

def _activation_to_string(activation):
	res = ''
	if activation is tf.nn.relu:
		res = 'R'
	if activation is tf.identity:
		res = 'I'
	if activation is tf.nn.sigmoid:
		res = 'S'
	if activation is tf.nn.elu:
		res = 'E'
	if activation is tf.nn.selu:
		res = 'SE'
	return res


def selu(x):
	with tf.variable_scope('selu'):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def elu(x):
	with tf.variable_scope('elu'):
		return tf.where(x>=0.0, x, tf.exp(x)-1)