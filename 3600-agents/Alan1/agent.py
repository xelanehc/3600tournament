from collections.abc import Callable
from collections import deque
from typing import List, Optional, Tuple
import time

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE,
    loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board

from .rat_belief import RatBelief
from .negamax import Negamax

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

_TIME_BUFFER = 0.15   # always keep this many seconds in reserve


# ═══════════════════════════════════════════════════════════════════════════════
#  Move generation (fallback only)
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_moves(board: Board, search_loc=None, search_ev=0.0) -> List[Move]:
    raw = board.get_valid_moves(exclude_search=True)
    result = list(raw)

    # Add exactly one search move if meaningful
    if search_loc is not None and search_ev > 0.1:
        result.append(Move.search(search_loc))

    # Order moves
    def key(m):
        if m.move_type == MoveType.CARPET:
            return 300 + CARPET_POINTS_TABLE.get(m.roll_length, 0)
        if m.move_type == MoveType.PRIME:
            return 200
        if m.move_type == MoveType.SEARCH:
            return 150 + search_ev
        return 0

    result.sort(key=key, reverse=True)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  Player Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:
    """
    Negamax + HMM agent with position tracking.

    Each turn:
    1. Update HMM with sensor data and any search results.
    2. If rat belief gives search EV above a dynamic threshold → Search.
    3. Otherwise → run negamax with iterative deepening and quiescence.
    """

    def __init__(self, board_state, transition_matrix=None,
                 time_left: Callable = None):
        self.belief  = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.engine  = Negamax(self.belief)
        self.turn    = 0
        self._prev_search = (None, False)
        self.pos_history = deque(maxlen=6)  # Track recent positions for repetition detection

    # ── commentate ──────────────────────────────────────────────────────

    def commentate(self) -> str:
        if self.belief is None:
            return "No belief."
        top = self.belief.top_n(3)
        parts = [f"{loc}:{p:.3f}" for loc, p in top]
        return f"Turn {self.turn} | rat top: {', '.join(parts)} | nodes: {self.engine.nodes}"

    # ── play ─────────────────────────────────────────────────────────────

    def play(self, board_state: Board, sensor_data: Tuple,
             time_left: Callable) -> Move:
        self.turn += 1
        noise, dist = sensor_data
        current_loc = board_state.player_worker.get_location()
        self.pos_history.append(current_loc)

        # ── Update HMM ─────────────────────────────────────────────────
        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(current_loc, dist)

            # Opponent's last search
            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc is not None:
                self.belief.update_search(opp_loc, opp_hit)

            # Our own last search (don't double-count)
            my_loc, my_hit = board_state.player_search
            if my_loc is not None and my_loc != self._prev_search[0]:
                self.belief.update_search(my_loc, my_hit)
                self._prev_search = (my_loc, my_hit)

        return self._decide(board_state, time_left)

    # ── decision logic ───────────────────────────────────────────────────

    def _decide(self, board: Board, time_left: Callable) -> Move:
        turns_left  = board.player_worker.turns_left
        t_remaining = time_left() - _TIME_BUFFER

        if t_remaining <= 0:
            moves = _gen_moves(board)
            return moves[0] if moves else Move.search((0, 0))

        # ── Rat search decision ─────────────────────────────────────────
        # EV = 6p − 2; positive when p > 1/3.
        # We add a dynamic "patience" threshold that relaxes over the game:
        #   Turn 1:  need EV ≥ 2.0  (p ≥ 0.67)  — very confident only
        #   Turn 30: need EV ≥ 0.0  (p ≥ 0.33)  — any positive EV
        if self.belief is not None:
            search_loc, search_ev = self.belief.best_search_ev()
            turns_played = MAX_TURNS_PER_PLAYER - turns_left
            patience = max(0.0, 2.0 - turns_played * (2.0 / 30.0))
            if search_ev >= patience:
                return Move.search(search_loc)

        # ── Negamax ────────────────────────────────────────────────────
        # Budget: allocate more per-turn time for deeper exploration.
        if turns_left > 0:
            per_turn = t_remaining / turns_left
        else:
            per_turn = t_remaining
        budget = max(0.10, min(6.0, per_turn * 0.95))

        search_loc, search_ev = self.belief.best_search_ev() if self.belief else (None, 0.0)
        return self.engine.best_move(board, budget, self.pos_history, search_loc, search_ev)