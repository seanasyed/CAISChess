import tensorflow as tf

class NeuralNet(): 

	def __init__(self, inputData): 
		self.inputLayer = inputData # TODO create input layer to transform info from graph

	# Either 19 or 39 residual blocks. We can probably mess with that number
	# Each residual block applies the following modules sequentially to its input
	def residualBlock(self, inputData): # Residual blocks that add input back at the end via skip layer implementation

		conv2 = tf.layers.conv2d(
			inputs=inputData,
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

		batchNorm3 += inputData

		relu3 = tf.nn.relu(batchNorm3)

		return relu3

	def policyHead(self, inputData): 
		conv4 = tf.layers.conv2d(
			inputs=inputData,
			filters=2,
			kernel_size=[1,1],
			stride=1)

		batchNorm4 = tf.layers.batch_normalization(conv4)

		relu4 = tf.nn.relu(batchNorm4)

		# return fully connected layer
		return tf.contrib.layers.fully_connected(
			inputs=relu4, num_outputs=392)

	def valueHead(self, input): 
		conv5 = tf.layers.conv2d(
			inputs=input, 
			kernel_size=[1,1], 
			filters=1, 
			stride=1)

		batchNorm5 = tf.layers.batch_normalization(conv5)

		relu5 = tf.nn.relu(batchNorm5)

		fullyConnected = tf.contrib.layers.fully_connected(
			inputs=relu5, num_outputs=256)

		relu6 = tf.nn.relu(fullyConnected)

		fullyConnected = tf.contrib.layers.fully_connected(
			inputs=relu6, num_outputs=1)

		return tf.math.tanh(fullyConnected)



	
	def convolutionalBlock(self): # Beginning of the net that will invoke the residual blocks

		# Convolutional Block
		conv1 = tf.layers.conv2d(
			inputs=self.inputLayer,
			filters=256,
			kernel_size=[3,3],
			stride=1)

		batchNorm1 = tf.layers.batch_normalization(conv1)

		relu1 = tf.nn.relu(batchNorm1)

		return relu1

	def initNeuralNet(self, numResidualBlocks): 
		self.numResidualBlocks = numResidualBlocks

		convolutionalBlock = convolutionalBlock()

		for n in range (10): 
			residualBlock = residualBlock(convolutionalBlock)

		policy = policyHead(residualBlock)
		value = valueHead(residualBlock)





