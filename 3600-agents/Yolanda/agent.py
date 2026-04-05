from collections.abc import Callable
from typing import List, Set, Tuple
import random

import numpy as np

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
        if transition_matrix:
            self.T = np.array(transition_matrix)
        else:
            self.T = np.eye(64)
        
        self.reset_rat()

        self.prev_opp_search = (None, False)
        self.prev_my_search = (None, False)

        self.NOISE_PROBS = {
            enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
            enums.Cell.SPACE: (0.7, 0.15, 0.15),
            enums.Cell.PRIMED: (0.1, 0.8, 0.1),
            enums.Cell.CARPET: (0.1, 0.1, 0.8),
        }
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "GGs."

    def reset_rat(self):
        """
        Resets HMM belief distro when rat's caught and new one runs 1000 steps
        """
        self.belief = np.zeros(64)
        self.belief[0] = 1.0 # reset belief to empty matrix then set rat to position (0, 0)

        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = self.belief @ T_1000
    
    def get_dist_prob(self, est_dist: int, actual_dist: int):
        """
        Returns probability of est_dist given actual_dist based on assignment file probabilities
        """
        if actual_dist == 0:
            if est_dist == 0:
                return 0.82
            elif est_dist == 1:
                return 0.12
            elif est_dist == 2:
                return 0.06
            else:
                return 0.0
        else:
            diff = est_dist - actual_dist
            if diff == -1:
                return 0.12
            elif diff == 0:
                return 0.7
            elif diff == 1:
                return 0.12
            elif diff == 2:
                return 0.06
            else:
                return 0.0

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
        if board.opponent_search[1] and board.opponent_search != self.prev_opp_search:
            self.reset_rat()
        elif board.player_search[1] and board.player_search != self.prev_my_search:
            self.reset_rat()

        # update most recent searches
        self.prev_opp_search = board.opponent_search
        self.prev_my_search = board.player_search
        
        # step forward in HMM
        self.belief = self.belief @ self.T

        # incorporate sensor data into belief
        noise_enum, est_dist = sensor_data
        my_loc = board.player_worker.get_location()

        for i in range(64):
            x, y = i % 8, i // 8
            actual_dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)

            p_dist = self.get_dist_prob(est_dist, actual_dist)

            cell_type = board.get_cell((x, y))
            p_noise = self.NOISE_PROBS[cell_type[int(noise_enum)]]

            self.belief[i] *= (p_dist * p_noise)
        
        # normalize belief matrix
        belief_sum = np.sum(self.belief)
        if belief_sum > 0:
            self.belief /= belief_sum
        else:
            self.belief = np.ones(64) / 64.0
        
        best_rat_idx = np.argmax(self.belief)
        if self.belief[best_rat_idx] > 0.75:
            search_loc = (int(best_rat_idx % 8), int(best_rat_idx // 8))
            return move.Move.search(search_loc)
            
        pass
