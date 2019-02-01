import random

class RandomEngine():

	def select_move(self, board):
		''' 
		Takes the current state of the board and returns a Move object.
		'''

		moves = list(board.legal_moves)
		index = random.randint(0, len(moves))

		return moves[index]




