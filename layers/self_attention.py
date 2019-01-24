import tensorflow as tf


class SelfAttention(object):
    def __init__(self,num_heads=1,linear_key_dim=50,linear_value_dim=50,attention_dim=100):
        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.attention_dim = attention_dim

    def __call__(self,input_Q,input_K,input_V):
        Q,K,V = self._linear_projection(input_Q,input_K,input_V)
        Q,K,V = self._split_to_heads(Q,K,V)
        out = self._scaled_dot_product(Q,K,V) # out [batch_size, num_heads, dim]
        out = self._concat_heads(out)
        out = linear(out,self.attention_dim,"attention_dim")
        return out

    def _linear_projection(self, q, k, v):
        q = linear(q, self.linear_key_dim,"linear_q")
        k = linear(k, self.linear_key_dim,"linear_k")
        v = linear(v, self.linear_value_dim,"linear_v")
        return q, k, v

    def _split_to_heads(self,Q,K,V):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1, t_shape[1],num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        Qs = split_last_dimension_then_transpose(Q, self.num_heads, self.linear_key_dim)
        Ks = split_last_dimension_then_transpose(K, self.num_heads, self.linear_key_dim)
        Vs = split_last_dimension_then_transpose(V, self.num_heads, self.linear_value_dim)

        return Qs, Ks, Vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads
        ks = tf.transpose(ks, [0, 1, 3, 2])
        o1 = tf.matmul(qs, ks) #, transpose_b=True
        o2 = o1 / (key_dim_per_head**0.5)

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)

    def _concat_heads(self, output):
        tensor = tf.transpose(output, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        return tf.reshape(tensor, [-1, t_shape[1], num_heads * dim])

    
def linear(X,N,scope="linear"):
    last_dims = X.get_shape().as_list()[1:]
    with tf.variable_scope(scope):
        W = tf.get_variable("W",[last_dims[-1],N],dtype=tf.float32)
        o = tf.reshape(X,[-1,last_dims[-1]])
        o = tf.matmul(o,W)
        o = tf.reshape(o,[-1]+last_dims[:-1]+[N])
        return o
