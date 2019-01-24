class Layer(object):
	def __call__(self,*args,**kwargs):
		raise NotImplementedError("not implemented")
	def to_json(self,sess):
		raise NotImplementedError("not implemented")