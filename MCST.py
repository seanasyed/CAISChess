import config, math
from chess import Board
import random

class Node():

	def __init__(self, move, parent=None):
		self.move = move
		self.parent = parent
		self.children = []

		self.visit_count = 0          # N
		self.prior_probability = 0    # P
		self.total_action_value = 0   # W
		self.mean_action_value = 0    # Q

	def add_win(self, val):
		self.visit_count += 1
		self.total_action_value += val

	def add_child(self, child):
		self.children.append(child)

	def get_action_value(self, total_count):
		# Calculate U
		upper_confidence_bound = config.c_puct * self.prior_probability * (math.sqrt(total_count) / (1 + self.visit_count))
		# Update Q
		self.mean_action_value = self.total_action_value/self.visit_count

		# Calculate W
		return self.mean_action_value + upper_confidence_bound


class MCSearchTree():

	def __init__(self, board):
		self.board = board
		self.root_node = Node(None, None)
		self.total_count = 0

	def run_simulation(self, node): # TODO: Implement this with the neural network. For testing, might want to try with rollout
		return random.randint(0, 1), random.uniform(0.0, 1.0)

	def add_moves(self, node):

		if node is None:
			return

		if node.move is not None:
			self.add_moves(node.parent)
			self.board.push(node.move)

	def remove_moves(self, node):

		while node is not None:
			if node.move is not None:
				self.board.pop()
			node = node.parent

	def expand_node(self, node):

		self.add_moves(node)

		for move in self.board.legal_moves:
			node.add_child(Node(move, parent=node))

		self.remove_moves(node)

		return node

	def back_prop(self, node, value):
		
		self.total_count += 1

		while node != None:
			node.add_win(value)
			node = node.parent

	def traverse_tree(self, node):
		
		while len(node.children) > 0:
			node = max(node.children, key=lambda x: x.get_action_value(self.total_count))

		return node


	def perform_iterations(self, num_iterations):

		for i in range(num_iterations):

			# Step a
			node = self.traverse_tree(self.root_node)  

			# Step b
			node = self.expand_node(node)         

			count = 0
			value = 0
			for child in node.children:  # Run a simulation for each child
				probability, result = self.run_simulation(child)
				child.add_win(result)
				child.prior_probability = probability
				count += 1
				value += result

			# Step c
			self.back_prop(node, value)

	# Step d
	def select_move(self):  # TODO: Implement as described on page 24
		return max(self.root_node.children, key=lambda x: x.visit_count).move