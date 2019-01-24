import tensorflow as tf
from .utils import _activation_to_string

class UGRnnCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
               num_units,
               activation=tf.nn.relu,
               reuse=None,
               kernel_initializer=tf.contrib.layers.xavier_initializer(),
               bias_initializer=tf.contrib.layers.xavier_initializer(),
               name=None,
               dtype=None,
               **kwargs):
        super(UGRnnCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        self._num_units = num_units
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self,batch_size, dtype=tf.float32):
        shape = [batch_size,self.state_size]
        return tf.zeros(shape,dtype=dtype)

    def __call__(self, inputs, state):
        self.feature_size = inputs.get_shape().as_list()[-1]
        self.time_size = inputs.get_shape().as_list()[-2]
        with tf.variable_scope("UGRNN",reuse=tf.AUTO_REUSE):
            self._context_w = tf.get_variable("context_w",shape=[self.feature_size*2,self._num_units],dtype=self.dtype,
                                            initializer=self._kernel_initializer,trainable=True)
            self._context_b = tf.get_variable("context_b",shape=[self._num_units],dtype=self.dtype,
                                            initializer=self._bias_initializer,trainable=True)
            self._gate_w = tf.get_variable("gate_w",shape=[self.feature_size*2,self._num_units],dtype=self.dtype,
                                            initializer=self._kernel_initializer,trainable=True)
            self._gate_b = tf.get_variable("gate_b",shape=[self._num_units],dtype=self.dtype,
                                            initializer=self._bias_initializer,trainable=True)

            inp = tf.concat([inputs, state],1)
            c = self._activation(tf.matmul(inp,self._context_w) + self._context_b)
            gate = tf.matmul(inp,self._gate_w) + self._gate_b
            
            # Returns 0. if x < -2.5, 1. if x > 2.5. In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.
            #gate = tf.keras.backend.hard_sigmoid(gate) 
            gate = tf.nn.sigmoid(gate) 
            out = gate * state + (1-gate) * c
            return out, c

    def to_json(self,sess):
        curr_layer = {}
        W_context, b_context, W_gate, b_gate = sess.run([self._context_w, self._context_b,self._gate_w,self._gate_b])
        curr_layer['W_context'] = W_context.tolist()
        curr_layer['b_context'] = b_context.tolist()
        curr_layer['W_gate'] = W_gate.tolist()
        curr_layer['b_gate'] = b_gate.tolist()

        curr_layer['in_dim'] = W_context.shape[0]
        curr_layer['out_dim'] = W_context.shape[1]
        curr_layer['time_dim'] = self.time_size
        curr_layer['activation'] = _activation_to_string(self._activation)
        curr_layer['type'] = "UGRNN"
        return curr_layer

class SRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self,
               num_units,
               activation=tf.nn.relu,
               reuse=None,
               kernel_initializer=tf.contrib.layers.xavier_initializer(),
               bias_initializer=tf.contrib.layers.xavier_initializer(),
               name=None,
               dtype=None,
               **kwargs):
        super(SRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        self._num_units = num_units
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self,batch_size, dtype=tf.float32):
        shape = [batch_size,self.state_size]
        return tf.zeros(shape,dtype=dtype)

    def __call__(self, inputs, state):
        self.feature_size = inputs.get_shape().as_list()[-1]
        self.time_size = inputs.get_shape().as_list()[-2]

        last_c = state
        with tf.variable_scope("SRU",reuse=tf.AUTO_REUSE):
            b_init = tf.zeros_initializer()
            self._context_w = tf.get_variable("context_w",shape=[self.feature_size,self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)
            self._out_w = tf.get_variable("out_w",shape=[self.feature_size,self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)

            self._gate_f_w = tf.get_variable("gate_f_w",shape=[self.feature_size,self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)
            self._gate_f_v = tf.get_variable("gate_f_v",shape=[self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)
            self._gate_f_b = tf.get_variable("gate_f_b",shape=[self._num_units],dtype=tf.float32,
                                            initializer=b_init,trainable=True)

            self._gate_r_w = tf.get_variable("gate_r_w",shape=[self.feature_size,self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)
            self._gate_r_v = tf.get_variable("gate_f_v",shape=[self._num_units],dtype=tf.float32,
                                            initializer=self._kernel_initializer,trainable=True)
            self._gate_r_b = tf.get_variable("gate_r_b",shape=[self._num_units],dtype=tf.float32,
                                            initializer=b_init,trainable=True)

            # Returns 0. if x < -2.5, 1. if x > 2.5. In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.
            #f = tf.keras.backend.hard_sigmoid(tf.matmul(inputs,self._gate_f_w) + self._gate_f_v*last_c + self._gate_f_b)
            f = tf.nn.sigmoid(tf.matmul(inputs,self._gate_f_w) + self._gate_f_v*last_c + self._gate_f_b)
            c = f*last_c + (1-f)*self._activation(tf.matmul(inputs,self._context_w))

            #r = tf.keras.backend.hard_sigmoid(tf.matmul(inputs,self._gate_r_w) + self._gate_r_v*last_c + self._gate_r_b)
            r = tf.nn.sigmoid(tf.matmul(inputs,self._gate_r_w) + self._gate_r_v*last_c + self._gate_r_b)
            #alpha = tf.sqrt(1+tf.exp(self._gate_r_b)*2)
            #alpha = tf.sqrt([3.0])
            out = r*c + (1-r) * tf.matmul(inputs,self._out_w) #* alpha
            return out, c
    
    def to_json(self,sess):
        curr_layer = {}
        W_context, W_out, f_W_gate, f_v_gate, f_b_gate, \
             r_W_gate, r_v_gate,r_b_gate = sess.run([self._context_w, self._out_w,
                            self._gate_f_w,self._gate_f_v,self._gate_f_b,
                            self._gate_r_w,self._gate_r_v,self._gate_r_b])
        curr_layer['W_context'] = W_context.tolist()
        curr_layer['W_out'] = W_out.tolist()
        curr_layer['f_W_gate'] = f_W_gate.tolist()
        curr_layer['f_v_gate'] = f_v_gate.tolist()
        curr_layer['f_b_gate'] = f_b_gate.tolist()
        curr_layer['r_W_gate'] = r_W_gate.tolist()
        curr_layer['r_v_gate'] = r_v_gate.tolist()
        curr_layer['r_b_gate'] = r_b_gate.tolist()

        curr_layer['in_dim'] = W_context.shape[0]
        curr_layer['out_dim'] = W_context.shape[1]
        curr_layer['time_dim'] = self.time_size
        curr_layer['activation'] = _activation_to_string(self._activation)
        curr_layer['type'] = "SRURNN"
        return curr_layer