from collections.abc import Callable
from typing import List, Set, Tuple
import random

from game import board, move, enums


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        pass
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        moves = board.get_valid_moves()
        return random.choice(moves)
