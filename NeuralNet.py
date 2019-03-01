import tensorflow as tf

class NeuralNet(): 

	def __init__(self, inputData): 
		self.inputLayer = inputData # TODO create input layer to transform info from graph
	
	def combinationalBlock(self, labels, mode): # Beginning of the net that will invoke the residual blocks

		# Convolutional Block
		conv1 = tf.layers.conv2d(
			inputs=self.inputLayer,
			filters=256,
			kernel_size=[3,3],
			stride=1)

		batchNorm1 = tf.layers.batch_normalization(conv1)

		relu1 = tf.nn.relu(batchNorm1)

		#TODO Invoke residual blocks

	# Either 19 or 39 residual blocks. We can probably mess with that number
	# Each residual block applies the following modules sequentially to its input
	def residualBlock(self, input): # TODO Residual blocks that add input back at the end via skip layer implementation
		pass



