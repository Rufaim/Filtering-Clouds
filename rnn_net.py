import tensorflow as tf
import numpy as np
import json

from layers import Dense

class SimpleRNN(object):
	def __init__(self,rnn_cell,overstructure,seq_len=5,feature_len=28,learning_rate=0.001,use_rnn_cell=True):
		self._rnn_cell = rnn_cell
		self._overstructure = overstructure
		self.learning_rate = learning_rate
		self._seq_len = seq_len
		self._feature_len = feature_len
		self._use_rnn_cell = use_rnn_cell

		self._build_ph()
		self._build_net()
		self._build_output()
		self._build_loss()
		self._build_optimizer()

		self.init = tf.global_variables_initializer()

	def _build_ph(self):
		self.input = tf.placeholder(tf.float32,[None,self._seq_len,self._feature_len])
		self.target = tf.placeholder(tf.float32,[None,1])
		self.weigth = tf.placeholder(tf.float32,[None,1])
		self.learning_rate_ph = tf.placeholder(tf.float32,[])

	def _build_net(self):
		if self._use_rnn_cell:
			out, _ = tf.nn.dynamic_rnn(self._rnn_cell, self.input,time_major=False, dtype=tf.float32)
			#out BxTxC
			out = out[:,-1]
		#out = Dense(56)(tf.reshape(self.input,[-1,self._seq_len*self._feature_len]))
		out = self.input
		for i,l in enumerate(self._overstructure):
			out = l(out,"layer_{}".format(i))
		
		self._last_layer = Dense(1)
		self.logit = self._last_layer(out,"out")

	def _build_output(self):
		self.out = tf.nn.sigmoid(self.logit)

	def _build_loss(self):
		loss = self.weigth*tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,logits=self.logit)

		self.loss = tf.reduce_sum(loss)
		self.loss_median = tf.contrib.distributions.percentile(loss, 50)

		self.accuracy= tf.reduce_mean(tf.cast(tf.equal(tf.round(self.out), self.target), dtype=tf.float32))

	def _build_optimizer(self):
		self.global_step = tf.Variable(0,trainable=False)
		self.learning_rate = tf.Variable(self.learning_rate,trainable=False)
		self.assign_LR = self.learning_rate.assign(self.learning_rate_ph)
		self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)

	def initialize(self,sess):
		self.sess = sess
		self.sess.run(self.init)

	def predict(self,X):
		return self.sess.run(self.out, feed_dict={
						self.input: np.reshape(X,(-1,self._seq_len,self._feature_len))
						})
	def train_step(self,X,Y,W=None):
		if W is None:
			W = np.ones_like(Y)
		self.sess.run(self.opt,feed_dict={
					self.input : np.reshape(X,(-1,self._seq_len,self._feature_len)),
					self.target : np.reshape(Y,(-1,1)),
					self.weigth : np.reshape(W,(-1,1)),
					})
	def get_loss(self,X,Y,W=None):
		if W is None:
			W = np.ones_like(Y)
		return self.sess.run([self.accuracy,self.loss, self.loss_median],feed_dict={
					self.input : np.reshape(X,(-1,self._seq_len,self._feature_len)),
					self.target : np.reshape(Y,(-1,1)),
					self.weigth : np.reshape(W,(-1,1)),
					})
	def set_learning_rate(self,lr):
		self.sess.run(self.assign_LR,feed_dict={self.learning_rate_ph: lr})
	def get_learning_rate(self):
		return self.sess.run(self.learning_rate)
	def get_global_step(self):
		return self.sess.run(self.global_step)

	def to_json(self,filename):
		print('Dumping network to file ' + filename)
		res = {"layer0":self._rnn_cell.to_json(self.sess)}
		for i,layer in enumerate(self._overstructure+[self._last_layer]):
			layer_name = "layer"+str(i+1)
			curr_layer = layer.to_json(self.sess)
			res[layer_name] = curr_layer
		res[layer_name]["activation"] = 'S'
		res["parameters"] = {"use_abs":True,"is_rnn":True,"use_last_rnn_out":True}
		with open(filename, 'w') as f:
			json.dump(res, f) 