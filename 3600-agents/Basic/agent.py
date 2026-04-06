from collections.abc import Callable
from typing import List, Set, Tuple
import random
import time
import numpy as np

from game import board, move, enums


class PlayerAgent:
    """
    Tournament-optimized Agent for CS3600 Carpet Game.
    Features an HMM for Rat Tracking and Iterative Deepening Alpha-Beta Minimax.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        """
        Initializes the agent, loading the transition matrix and setting the initial rat belief.
        """
        # 1. HMM Initialization
        if transition_matrix is not None:
            self.T = np.array(transition_matrix)
        else:
            self.T = np.eye(64) # Fallback if missing
            
        self._reset_rat()
        
        # History tracking to know when a rat has been caught
        self.prev_opp_search = (None, False)
        self.prev_my_search = (None, False)

        # Precompute noise probabilities for fast lookup
        self.NOISE_PROBS = {
            enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
            enums.Cell.SPACE: (0.7, 0.15, 0.15),
            enums.Cell.PRIMED: (0.1, 0.8, 0.1),
            enums.Cell.CARPET: (0.1, 0.1, 0.8),
        }

    def _reset_rat(self):
        """
        Resets the HMM belief distribution when a rat is caught and allowed a 1000-move headstart.
        """
        self.belief = np.zeros(64)
        self.belief[0] = 1.0 # Rat always starts at (0,0)
        
        # Simulate 1000 steps instantly using matrix exponentiation
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = self.belief @ T_1000

    def _get_dist_prob(self, est_dist: int, actual_dist: int) -> float:
        """
        Returns the probability of the sensor reporting est_dist given actual_dist.
        Based on the exact probabilities defined in rat.py.
        """
        if actual_dist == 0:
            if est_dist == 0: return 0.82   # (-1 offsets to 0) + (0 is correct)
            elif est_dist == 1: return 0.12 # (+1 error)
            elif est_dist == 2: return 0.06 # (+2 error)
            return 0.0
        else:
            diff = est_dist - actual_dist
            if diff == -1: return 0.12
            elif diff == 0: return 0.70
            elif diff == 1: return 0.12
            elif diff == 2: return 0.06
            return 0.0

    def commentate(self):
        return "You've been caught in my web (or rather, my carpet)."

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        start_time = time.time()
        
        # ----------------------------------------------------------------------
        # PART 1: THE RAT TRACKER (Hidden Markov Model)
        # ----------------------------------------------------------------------
        
        # Check if rat was caught since our last update
        if board.opponent_search[1] and board.opponent_search != self.prev_opp_search:
            self._reset_rat()
        elif board.player_search[1] and board.player_search != self.prev_my_search:
            self._reset_rat()
            
        self.prev_opp_search = board.opponent_search
        self.prev_my_search = board.player_search
        
        # A) Predict Step: Update belief based on transition matrix
        self.belief = self.belief @ self.T
        
        # B) Update Step: Fuse sensor data (noise, distance) into belief
        noise_enum, est_dist = sensor_data
        my_loc = board.player_worker.get_location()
        
        for i in range(64):
            x, y = i % 8, i // 8
            actual_dist = abs(my_loc[0] - x) + abs(my_loc[1] - y)
            
            p_dist = self._get_dist_prob(est_dist, actual_dist)
            
            cell_type = board.get_cell((x, y))
            p_noise = self.NOISE_PROBS[cell_type][int(noise_enum)]
            
            self.belief[i] *= (p_dist * p_noise)
            
        # Normalize the belief matrix
        belief_sum = np.sum(self.belief)
        if belief_sum > 0:
            self.belief /= belief_sum
        else:
            self.belief = np.ones(64) / 64.0 # Fallback in case of absolute zeroing
            
        # ----------------------------------------------------------------------
        # PART 2: THE ACTION EVALUATOR 
        # ----------------------------------------------------------------------
        
        # First, check if we are highly confident in the rat's location.
        # Capturing the rat yields 4 points. The penalty is 2 points. 
        # An expected value > 0 requires confidence > ~0.33, but practically 
        # we want to be much more certain so we don't waste a turn.
        best_rat_idx = np.argmax(self.belief)
        if self.belief[best_rat_idx] > 0.75:
            search_loc = (int(best_rat_idx % 8), int(best_rat_idx // 8))
            return move.Move.search(search_loc)

        # Otherwise, fall back to spatial movement planning.
        valid_moves = board.get_valid_moves(exclude_search=True)
        if not valid_moves:
            # Absolute fallback if trapped
            return move.Move.search((0,0))
            
        # Dynamic Time Management: Don't spend all 4 minutes on turn 1.
        turns_remaining = max(1, board.player_worker.turns_left)
        allocated_time = min(time_left() / turns_remaining, 4.5) 
        
        best_move = random.choice(valid_moves) # Fallback to random
        
        # Iterative Deepening
        try:
            for depth in range(1, 5): 
                eval_score, potential_move = self._minimax(
                    board_state=board,
                    depth=depth, 
                    alpha=float('-inf'), 
                    beta=float('inf'), 
                    is_maximizing=True, 
                    start_time=start_time, 
                    time_budget=allocated_time
                )
                if potential_move is not None:
                    best_move = potential_move
                
                # Exit early if we are approaching our allocated time for this turn
                if time.time() - start_time > allocated_time * 0.85:
                    break
        except TimeoutError:
            pass # Keep the best move found in the previous completed depth
            
        return best_move

    def _minimax(self, board_state, depth, alpha, beta, is_maximizing, start_time, time_budget):
        """
        Alpha-Beta Minimax search tree to evaluate optimal moves.
        """
        if time.time() - start_time > time_budget:
            raise TimeoutError()
            
        if depth == 0 or board_state.is_game_over():
            return self._evaluate_board(board_state, is_maximizing), None
            
        if is_maximizing:
            max_eval = float('-inf')
            best_move = None
            valid_moves = board_state.get_valid_moves(enemy=False, exclude_search=True)
            
            # Simple move ordering to prune faster (Carpet > Prime > Plain)
            valid_moves.sort(key=lambda m: m.move_type, reverse=True)
            
            for m in valid_moves:
                next_board = board_state.forecast_move(m)
                if next_board is None: continue
                
                # Flip the perspective so the enemy can take their turn
                next_board.reverse_perspective()
                
                eval_score, _ = self._minimax(next_board, depth - 1, alpha, beta, False, start_time, time_budget)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = m
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
            
        else:
            min_eval = float('inf')
            best_move = None
            
            # Because perspective was reversed by the max node, 'enemy=False' gets the opponent's moves
            valid_moves = board_state.get_valid_moves(enemy=False, exclude_search=True)
            valid_moves.sort(key=lambda m: m.move_type, reverse=True)

            for m in valid_moves:
                next_board = board_state.forecast_move(m)
                if next_board is None: continue
                
                next_board.reverse_perspective()
                
                eval_score, _ = self._minimax(next_board, depth - 1, alpha, beta, True, start_time, time_budget)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = m
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _evaluate_board(self, board_state, is_maximizing) -> float:
        """
        Heuristic evaluation function. Prioritizes point advantage and long carpet potentials.
        """
        my_worker = board_state.player_worker if is_maximizing else board_state.opponent_worker
        opp_worker = board_state.opponent_worker if is_maximizing else board_state.player_worker
        
        # Base weight heavily on actual points scored
        score = (my_worker.points - opp_worker.points) * 100
        
        # Add future potential based on available moves
        my_moves = board_state.get_valid_moves(enemy=not is_maximizing, exclude_search=True)
        for m in my_moves:
            if m.move_type == enums.MoveType.PRIME:
                score += 2
            elif m.move_type == enums.MoveType.CARPET:
                score += (m.roll_length * 5) # Highly reward long carpets
                
        # Deduct future potential of the opponent
        opp_moves = board_state.get_valid_moves(enemy=is_maximizing, exclude_search=True)
        for m in opp_moves:
            if m.move_type == enums.MoveType.PRIME:
                score -= 2
            elif m.move_type == enums.MoveType.CARPET:
                score -= (m.roll_length * 5)
                
        return float(score)