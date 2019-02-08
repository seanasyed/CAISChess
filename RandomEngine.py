import random

class RandomEngine():

	def select_move(self, board):
		''' 
		Takes the current state of the board and returns a Move object.
		'''

		moves = list(board.legal_moves)
		
		if len(moves) == 0:
			return None


		try:
			index = random.randint(0, len(moves) - 1)
			print(index, len(moves))
		except:
			raise '{} {}'.format(index, len(moves))
		return moves[index]




