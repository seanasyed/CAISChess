import tensorflow as tf

class NeuralNet(): 

	def __init__(self, inputData): 
		self.inputLayer = tf.reshape(inputData) #TODO create input layer to transform info from graph
	
	def rlModel(self, labels, mode):
		conv1 = tf.layers.conv2d(
			inputs=self.inputLayer,
			filters=256,
			kernel_size=[3,3],
			stride=1)