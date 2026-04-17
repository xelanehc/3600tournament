from collections.abc import Callable
from collections import deque
from typing import List, Optional, Tuple
import time

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board

from .negamax import Negamax
from .rat_belief import RatBelief

_TIME_BUFFER = 0.12
_SEARCH_COOLDOWN = 2
_SEARCH_HIGH_CONF = 1.65
_SEARCH_BASE_GATE = 1.5
_ENDGAME_RESERVE_FRAC = 0.20
_ENDGAME_THRESHOLD = 10
_MIN_BUDGET = 0.08


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


def _best_non_search_opportunity(board: Board) -> float:
    moves = board.get_valid_moves(exclude_search=True)
    best = 0.0
    for m in moves:
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
            if pts > best:
                best = float(pts)
        elif m.move_type == MoveType.PRIME:
            if 0.8 > best:
                best = 0.8
        elif m.move_type == MoveType.PLAIN:
            if 0.4 > best:
                best = 0.4
    return best


class PlayerAgent:
    """Alan5: enhanced search, eval, rat belief, and time management."""

    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self.belief = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.engine = Negamax()
        self.turn = 0
        self.pos_history = deque(maxlen=6)
        self.last_nodes = 0
        self._search_cooldown_counter = 0
        self._prev_search = (None, None)

    def commentate(self) -> str:
        if self.belief is None:
            return f"Turn {self.turn} | depth: {self.engine.last_depth} | nodes: {self.last_nodes}"
        top = self.belief.top_n(3)
        top_str = ", ".join(f"{loc}:{p:.2f}" for loc, p in top)
        return f"Turn {self.turn} | depth: {self.engine.last_depth} | nodes: {self.last_nodes} | rat: {top_str}"

    def _should_search_now(self, search_ev: float, board_state: Board) -> bool:
        if self._search_cooldown_counter > 0:
            return False

        turns_left = board_state.player_worker.turns_left
        frac_done = 1.0 - turns_left / MAX_TURNS_PER_PLAYER if MAX_TURNS_PER_PLAYER > 0 else 1.0

        if search_ev >= _SEARCH_HIGH_CONF:
            return True

        phase_threshold = _SEARCH_BASE_GATE * (1.0 - frac_done)
        opp_cost = _best_non_search_opportunity(board_state)
        min_required = max(phase_threshold, opp_cost + 0.2)

        return search_ev >= min_required

    def play(self, board_state: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turn += 1
        if self._search_cooldown_counter > 0:
            self._search_cooldown_counter -= 1

        noise, dist = sensor_data
        current_loc = board_state.player_worker.get_location()
        self.pos_history.append(current_loc)

        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(current_loc, dist)

            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc is not None:
                self.belief.update_search(opp_loc, opp_hit, is_opponent=True)

            my_loc, my_hit = board_state.player_search
            if my_loc is not None and my_loc != self._prev_search[0]:
                self.belief.update_search(my_loc, my_hit, is_opponent=False)
                self._prev_search = (my_loc, my_hit)

        turns_left = board_state.player_worker.turns_left
        t_remaining = time_left() - _TIME_BUFFER
        if t_remaining <= 0:
            moves = _gen_moves(board_state)
            return moves[0] if moves else Move.search((0, 0))

        search_loc, search_ev = (None, 0.0)
        if self.belief is not None:
            search_loc, search_ev = self.belief.best_search_ev()
            if self._should_search_now(search_ev, board_state):
                self._search_cooldown_counter = _SEARCH_COOLDOWN
                return Move.search(search_loc)

        if turns_left <= _ENDGAME_THRESHOLD:
            spendable = t_remaining * (1.0 - _ENDGAME_RESERVE_FRAC)
        else:
            spendable = t_remaining

        if turns_left > 30:
            time_frac = 0.80
        elif turns_left > 24:
            time_frac = 1.30
        elif turns_left > 16:
            time_frac = 1.60
        elif turns_left > 8:
            time_frac = 1.30
        else:
            time_frac = 1.00

        if turns_left > 0:
            per_turn = spendable / turns_left
        else:
            per_turn = spendable
        budget = max(_MIN_BUDGET, per_turn * time_frac)

        move = self.engine.best_move(board_state, budget, self.pos_history,
                                     search_loc, search_ev)
        self.last_nodes = self.engine.nodes
        return move
