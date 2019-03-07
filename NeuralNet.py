import tensorflow as tf

class NeuralNet(): 

	def __init__(self, inputData): 
		self.inputLayer = inputData # TODO create input layer to transform info from graph

	# Either 19 or 39 residual blocks. We can probably mess with that number
	# Each residual block applies the following modules sequentially to its input
	def residualBlock(self, input, numBlocks): # Residual blocks that add input back at the end via skip layer implementation

		conv2 = tf.layers.conv2d(
			inputs=input,
			filters=256,
			kernel_size=[3,3],
			stride=1)

		batchNorm2 = tf.layers.batch_normalization(conv2)

		relu2 = tf.nn.relu(batchNorm2)

		conv3 = tf.layers.conv2d(
			inputs=relu2,
			filters=256,
			kernel_size=[3,3],
			stride=1)

		batchNorm3 = tf.layers.batch_normalization(conv3)

		batchNorm3 += numBlocks

		relu3 = tf.nn.relu(batchNorm3)

		if numBlocks == 1: 
			return relu3
		else:
			return residualBlock(relu3, numBlocks - 1)

	def policyHead(self, input): 
		conv4 = tf.layers.conv2d(
			inputs=input,
			filters=2,
			kernel_size=[1,1],
			stride=1)

		batchNorm4 = tf.layers.batch_normalization(conv4)

		relu4 = tf.nn.relu(batchNorm4)

		# TODO return fully connected layer

	def valueHead(self, input): 
		pass
	
	def combinationalBlock(self, labels, mode): # Beginning of the net that will invoke the residual blocks

		# Convolutional Block
		conv1 = tf.layers.conv2d(
			inputs=self.inputLayer,
			filters=256,
			kernel_size=[3,3],
			stride=1)

		batchNorm1 = tf.layers.batch_normalization(conv1)

		relu1 = tf.nn.relu(batchNorm1)

		residual = residualBlock(relu1, 13)

		policy = policyHead(residual)

		value = valueHead(residual)

	







