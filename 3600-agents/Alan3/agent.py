from collections.abc import Callable
from collections import deque
from typing import List, Optional, Tuple
import time

from game.enums import Direction, MoveType, Cell, BOARD_SIZE, CARPET_POINTS_TABLE, loc_after_direction, MAX_TURNS_PER_PLAYER
from game.move import Move
from game.board import Board

from .negamax import Negamax
from .rat_belief import RatBelief

_TIME_BUFFER = 0.15


def _gen_moves(board: Board) -> List[Move]:
    raw = board.get_valid_moves(exclude_search=True)
    result = list(raw)

    def key(m):
        if m.move_type == MoveType.CARPET:
            return 300 + CARPET_POINTS_TABLE.get(m.roll_length, 0)
        if m.move_type == MoveType.PRIME:
            return 200
        return 0

    result.sort(key=key, reverse=True)
    return result


class PlayerAgent:
    """Simple score-difference agent with alpha-beta negamax search and rat belief."""

    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self.belief = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.engine = Negamax()
        self.turn = 0
        self.pos_history = deque(maxlen=6)
        self.last_nodes = 0

    def commentate(self) -> str:
        if self.belief is None:
            return f"Turn {self.turn} | depth: {self.engine.last_depth} | nodes: {self.last_nodes}"
        top = self.belief.top_n(3)
        top_str = ", ".join(f"{loc}:{p:.2f}" for loc, p in top)
        return f"Turn {self.turn} | depth: {self.engine.last_depth} | nodes: {self.last_nodes} | rat: {top_str}"

    def play(self, board_state: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turn += 1
        noise, dist = sensor_data
        current_loc = board_state.player_worker.get_location()
        self.pos_history.append(current_loc)

        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(current_loc, dist)

            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc is not None:
                self.belief.update_search(opp_loc, opp_hit)

            my_loc, my_hit = board_state.player_search
            if my_loc is not None and my_loc != getattr(self, '_prev_search', (None, None))[0]:
                self.belief.update_search(my_loc, my_hit)
                self._prev_search = (my_loc, my_hit)

        turns_left = board_state.player_worker.turns_left
        t_remaining = time_left() - _TIME_BUFFER
        if t_remaining <= 0:
            moves = _gen_moves(board_state)
            return moves[0] if moves else Move.search((0, 0))

        search_loc, search_ev = (None, 0.0)
        if self.belief is not None:
            search_loc, raw_search_ev, search_confidence = self.belief.best_search_ev()
            turns_played = MAX_TURNS_PER_PLAYER - turns_left
            # Apply an early-game penalty to rat searching (discourage in early turns, encourage later)
            # Early game penalty is high when turns_played is low, decreases as game progresses
            early_game_penalty = (MAX_TURNS_PER_PLAYER - turns_played) * 0.05
            search_ev = max(0.0, raw_search_ev - early_game_penalty)
            # Only consider searching if we have reasonable confidence (>50%) that the rat is there
            min_confidence = 0.4
            # Patience decreases as game progresses (become more eager to search later)
            patience = max(0.0, (MAX_TURNS_PER_PLAYER - turns_played) * (2.0 / 30.0))
            if search_ev >= patience and search_confidence > min_confidence:
                return Move.search(search_loc)

        # Use more time in the middle of the game, and consume the remaining
        # time aggressively while keeping a safe upper bound.
        if turns_left > 30:
            time_frac = 0.7
        elif turns_left > 24:
            time_frac = 1.1
        elif turns_left > 16:
            time_frac = 1.5
        elif turns_left > 8:
            time_frac = 1.2
        else:
            time_frac = 0.8

        if turns_left > 0:
            per_turn = t_remaining / turns_left
        else:
            per_turn = t_remaining
        budget = max(0.05, per_turn * time_frac)

        move = self.engine.best_move(board_state, budget, self.pos_history, search_loc, search_ev)
        self.last_nodes = self.engine.nodes
        return move
