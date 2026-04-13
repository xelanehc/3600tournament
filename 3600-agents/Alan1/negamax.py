"""
Negamax search engine with alpha-beta pruning and quiescence.

This module implements the Negamax class for game tree search with
iterative deepening and position evaluation.
"""

from collections import deque
from typing import List, Optional, Tuple
import time

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE,
    loc_after_direction,
)
from game.move import Move
from game.board import Board

from .rat_belief import RatBelief

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

ALL_DIRS = list(Direction)
_MAX_DEPTH = 12     # iterative deepening ceiling (increased for deeper exploration)
_MIN_SECS = 0.01    # don't start a new depth if less than this left

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _dist(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# ═══════════════════════════════════════════════════════════════════════════════
#  Heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def _lane_access(board: Board, pos: Tuple[int,int], opponent_pos: Tuple[int,int]) -> float:
    """
    Measures how open the area is around `pos` by counting
    how many steps you can move before hitting a wall or block.
    Does NOT try to estimate carpet scoring.
    """
    openness = 0.0

    for d in ALL_DIRS:
        cur = pos
        steps = 0

        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur) or board.is_cell_blocked(cur):
                break

            # Opponent contesting doesn't matter for openness
            if board.get_cell(cur) in (Cell.SPACE, Cell.PRIMED, Cell.CARPET):
                steps += 1
                continue

            break

        # Reward directions with more freedom
        openness += steps ** 0.5   # diminishing returns

    return openness

def _contested_carpet_potential(board: Board, player_pos, opponent_pos):
    total_val = 0.0

    for d in ALL_DIRS:
        cur = player_pos
        primed = 0
        space_run = 0

        # Step 1: Count PRIMED cells
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur) or board.is_cell_blocked(cur):
                break

            c = board.get_cell(cur)

            if c == Cell.PRIMED:
                primed += 1
                continue

            # Step 2: Count SPACE cells (open row)
            if c == Cell.SPACE:
                # Opponent contest check
                d_me  = _dist(player_pos, cur)
                d_opp = _dist(opponent_pos, cur)
                if d_opp <= d_me:
                    break

                space_run += 1
                # Continue scanning forward for more SPACE
                continue

            break

        # --- Immediate carpet value ---
        if primed >= 2:
            total_val += CARPET_POINTS_TABLE.get(primed, 0) * (1.0 + 0.1 * primed)

        # --- Future open‑row SPACE potential ---
        if space_run >= 2:
            # Total possible chain = primed + future SPACE
            future_len = primed + space_run
            future_len = min(future_len, 7)

            # Discount based on how many priming turns needed
            discount = 1.0 / (1.0 + space_run)

            total_val += CARPET_POINTS_TABLE.get(future_len, 0) * 0.4 * discount

        # --- Single SPACE extension (your original logic) ---
        elif space_run == 1:
            potential_len = primed + 1
            d_me = _dist(player_pos, cur)
            d_opp = _dist(opponent_pos, cur)
            if d_opp > d_me:
                closeness = max(0.1, (d_opp - d_me))
                weight = closeness / (1.0 + d_me)
                total_val += CARPET_POINTS_TABLE.get(min(potential_len, 7), 0) * weight

    return total_val

PHASE_WEIGHTS = {
    "opening":  {"score": 0.4, "potential": 0.7, "rat_ev": 0.1, "lane_access": 0.35},
    "midgame":  {"score": 1.0, "potential": 0.9,  "rat_ev": 0.35, "lane_access": 0.2},
    "endgame":  {"score": 1.6, "potential": 1.3,  "rat_ev": 0.5,  "lane_access": 0.1},
}

def evaluate(board: Board, belief: Optional['RatBelief'], pos_history: Optional[deque] = None) -> float:
    pw = board.player_worker
    ow = board.opponent_worker

    ppos = pw.get_location()
    opos = ow.get_location()

    # Phase
    turn = board.turn_count
    phase = "opening" if turn < 10 else "midgame" if turn < 30 else "endgame"
    W = PHASE_WEIGHTS[phase]

    # Features

    # 1. Score difference
    score_delta = float(pw.get_points() - ow.get_points())

    # 2. Contested carpet potential
    my_pot  = _contested_carpet_potential(board, ppos, opos)
    opp_pot = _contested_carpet_potential(board, opos, ppos)
    pot_delta = my_pot - (opp_pot * 1.2)  # Slightly penalize opponent potential

    # 3. Rat EV contribution
    rat_ev = belief.rat_ev() if belief is not None else 0.0

    # Tempo penalty: searching costs a move
    tempo_penalty = 0.0
    if rat_ev > 0:
        tempo_penalty = 0.3 * rat_ev + 0.1 * opp_pot

    # 4. Centrality
    lane_delta = _lane_access(board, ppos, opos) - _lane_access(board, opos, ppos)

    # 5. Repetition penalty (avoid cycles)
    repetition_penalty = 0.0
    if pos_history is not None:
        for i, prev_pos in enumerate(reversed(pos_history)):
            if ppos == prev_pos:
                repetition_penalty -= 1.5 / (i + 1)

    return (
        W["score"]      * score_delta +
        W["potential"]  * pot_delta +
        W.get("rat_ev", 0) * rat_ev - tempo_penalty +
        W["lane_access"] * lane_delta +
        repetition_penalty
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Move generation
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_moves(board: Board, search_loc=None, search_ev=0.0) -> List[Move]:
    """
    Generate all valid moves including:
      - normal moves (MOVE, PRIME, CARPET)
      - the single best SEARCH move (if valuable)
    """
    raw = board.get_valid_moves(exclude_search=True)
    result = list(raw)

    # Add the best search move if it exists and is meaningful
    if search_loc is not None and search_ev > 0.1:
        result.append(Move.search(search_loc))

    # Order: carpet (descending length) > prime > search > plain
    def key(m):
        if m.move_type == MoveType.CARPET:
            return 300 + CARPET_POINTS_TABLE.get(m.roll_length, 0)
        if m.move_type == MoveType.PRIME:
            return 200
        if m.move_type == MoveType.SEARCH:
            return 150 + search_ev  # prioritize high-value searches
        return 0

    result.sort(key=key, reverse=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Negamax with alpha-beta and iterative deepening
# ═══════════════════════════════════════════════════════════════════════════════

INF = float('inf')

class _Timeout(Exception):
    pass

class Negamax:
    """
    Negamax with alpha-beta pruning and quiescence.

    Convention: every call returns a value from the perspective of
    the CURRENT player (player_worker on the board passed in).
    """

    def __init__(self, belief: Optional[RatBelief]):
        self.belief   = belief
        self.nodes    = 0
        self._dead    = 0.0

    # ── Public entry ────────────────────────────────────────────────────

    def best_move(self, board, budget, pos_history, search_loc, search_ev):
        self.search_loc = search_loc
        self.search_ev  = search_ev
        self._dead = time.perf_counter() + budget
        moves = _gen_moves(board, search_loc, search_ev)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))

        best = moves[0]
        for depth in range(1, _MAX_DEPTH + 1):
            if self._dead - time.perf_counter() < _MIN_SECS:
                break
            self.nodes = 0
            try:
                _, mv = self._negamax(board, depth, -INF, INF, pos_history or deque())
                if mv is not None:
                    best = mv
            except _Timeout:
                break   # keep best from last complete depth
        return best

    # ── Core negamax ─────────────────────────────────────────────────────

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float,
                 pos_history: deque) -> Tuple[float, Optional[Move]]:

        if time.perf_counter() >= self._dead:
            raise _Timeout()

        self.nodes += 1

        if board.is_game_over():
            val = evaluate(board, self.belief, pos_history)
            return val, None

        if depth == 0:
            # Use quiescence for leaf evaluation
            return self._quiescence(board, alpha, beta, pos_history), None
        
        moves = _gen_moves(board, self.search_loc, self.search_ev)
        if not moves:
            return evaluate(board, self.belief, pos_history), None

        best_val = -INF
        best_move = None

        for m in moves:
            child = board.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()

            child_pos = child.player_worker.get_location()
            child_hist = pos_history.copy()
            child_hist.append(child_pos)
            if len(child_hist) > 6:
                child_hist.popleft()

            child_val, _ = self._negamax(child, depth - 1, -beta, -alpha, child_hist)
            val = -child_val

            if val > best_val:
                best_val = val
                best_move = m

            alpha = max(alpha, val)
            if alpha >= beta:
                break

        return best_val, best_move

    def _quiescence(self, board: Board, alpha: float, beta: float,
                    pos_history: deque) -> float:
        """Quiescence search: evaluate position more carefully at leaves."""
        stand_pat = evaluate(board, self.belief, pos_history)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        # Only search high-value carpet/prime moves
        moves = _gen_moves(board, self.search_loc, self.search_ev)
        for m in moves:
            if m.move_type not in (MoveType.CARPET, MoveType.PRIME):
                continue

            child = board.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()

            child_pos = child.player_worker.get_location()
            child_hist = pos_history.copy()
            child_hist.append(child_pos)
            if len(child_hist) > 6:
                child_hist.popleft()

            score = -self._quiescence(child, -beta, -alpha, child_hist)

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha