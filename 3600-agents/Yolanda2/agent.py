"""
Carpet-game tournament agent.

Key fixes over previous version
================================
1.  NEGAMAX (not split max/min): single recursive function, always from the
    current player's perspective.  Alpha-beta windows are negated and swapped
    correctly at every level.

2.  forecast_move returns a board from the MOVER'S perspective still.
    We call reverse_perspective() once after forecast so that the recursive
    call sees the board from the NEW current player's perspective.
    evaluate() is always from player_worker's POV → negation is trivial.

3.  No search moves inside the tree.  Searches are decided before calling
    negamax via the HMM threshold.  Mixing searches into the tree while
    using a static belief snapshot causes the tree to evaluate them
    incorrectly (the rat belief doesn't update mid-tree).

4.  Heuristic strongly rewards: real score delta, reachable carpet chains,
    open priming lanes — and penalises wasted/redundant plain moves by
    considering how many turns are needed to convert potential into points.

5.  Carpet of length 1 (= -1 pts) is filtered out of valid moves before
    any search.

6.  Time is allocated per-turn with a floor so early turns don't burn the
    whole budget.
"""

from collections.abc import Callable
from typing import List, Optional, Tuple
import time

import numpy as np

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE,
    loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

N = BOARD_SIZE * BOARD_SIZE
ALL_DIRS = list(Direction)

# P(noise_type | cell_type)  indices: 0=squeak, 1=scratch, 2=squeal
NOISE_PROBS = {
    Cell.BLOCKED: (0.5,  0.30, 0.20),
    Cell.SPACE:   (0.70, 0.15, 0.15),
    Cell.PRIMED:  (0.10, 0.80, 0.10),
    Cell.CARPET:  (0.10, 0.10, 0.80),
}
# P(observed = actual + offset)
DIST_ERR_P  = (0.12, 0.70, 0.12, 0.06)
DIST_ERR_OFF = (-1,   0,    1,    2)

# Search budget
_TIME_BUFFER  = 0.25   # always keep this many seconds in reserve
_MAX_DEPTH    = 8      # iterative deepening ceiling
_MIN_SECS     = 0.03   # don't start a new depth if less than this left

# Heuristic weights  (all in units of "carpet points")
W_SCORE      = 1.0    # actual score delta
W_POTENTIAL  = 0.45   # reachable carpet potential
W_RAT        = 0.60   # best rat-search EV contribution
W_CENTRALITY = 0.10   # favour positions closer to board centre


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _idx(loc: Tuple[int,int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]

def _loc(idx: int) -> Tuple[int,int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)

def _dist(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

_CENTRE = (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1)   # (3,3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Hidden Markov Model
# ═══════════════════════════════════════════════════════════════════════════════

class RatBelief:
    """
    Probability distribution over all 64 cells for the rat's position.

    Each player-turn, call in order:
        predict()                 # rat moves via T
        update_noise(board, n)    # condition on noise heard
        update_distance(pos, d)   # condition on reported distance
        update_search(loc, hit)   # for every search that occurred
    """

    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.b = np.full(N, 1.0 / N)   # uniform prior

    def predict(self):
        self.b = self.b @ self.T
        s = self.b.sum()
        if s > 0:
            self.b /= s

    def update_noise(self, board: Board, noise_type):
        ni = int(noise_type)
        for i in range(N):
            try:
                cell = board.get_cell(_loc(i))
            except Exception:
                cell = Cell.SPACE
            self.b[i] *= NOISE_PROBS.get(cell, NOISE_PROBS[Cell.SPACE])[ni]
        self._norm()

    def update_distance(self, pos: Tuple[int,int], observed: int):
        for i in range(N):
            actual = _dist(pos, _loc(i))
            p = 0.0
            for off, prob in zip(DIST_ERR_OFF, DIST_ERR_P):
                if max(0, actual + off) == observed:
                    p += prob
            self.b[i] *= p
        self._norm()

    def update_search(self, loc: Tuple[int,int], found: bool):
        if found:
            self.b[:] = 1.0 / N          # new rat → uniform
        else:
            self.b[_idx(loc)] = 0.0
            self._norm()

    def _norm(self):
        s = self.b.sum()
        if s > 1e-15:
            self.b /= s
        else:
            self.b[:] = 1.0 / N

    def best_search_ev(self) -> Tuple[Tuple[int,int], float]:
        """(location, EV) for single best search.  EV = 6p − 2."""
        i = int(np.argmax(self.b))
        p = float(self.b[i])
        return _loc(i), p * RAT_BONUS - (1.0 - p) * RAT_PENALTY

    def rat_ev(self) -> float:
        _, ev = self.best_search_ev()
        return ev

    def top_n(self, n=3):
        idx = np.argsort(self.b)[-n:][::-1]
        return [(_loc(int(i)), float(self.b[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def _primed_run_length(board: Board, start: Tuple[int,int],
                       direction: Direction,
                       p_pos: Tuple[int,int], o_pos: Tuple[int,int]) -> int:
    """
    Count consecutive PRIMED cells in `direction` from `start` (exclusive).
    Stops at board edge, BLOCKED, CARPET, SPACE, or either worker.
    """
    length = 0
    cur = start
    for _ in range(BOARD_SIZE - 1):
        cur = loc_after_direction(cur, direction)
        if not board.is_valid_cell(cur):
            break
        if cur == p_pos or cur == o_pos:
            break
        c = board.get_cell(cur)
        if c == Cell.PRIMED:
            length += 1
        else:
            break
    return length


def _carpet_potential(board: Board, pos: Tuple[int,int]) -> float:
    """
    Sum of carpet points immediately achievable (carpet roll) from pos,
    plus discounted credit for open lanes that could be primed.
    """
    p_pos = board.player_worker.get_location()
    o_pos = board.opponent_worker.get_location()
    total = 0.0
    for d in ALL_DIRS:
        run = _primed_run_length(board, pos, d, p_pos, o_pos)
        if run >= 2:
            total += CARPET_POINTS_TABLE.get(run, 0)
        elif run == 1:
            pass   # 1-carpet = -1 pts, don't count as potential
        # Credit for open SPACE lanes (future priming opportunity)
        cur = pos
        space_len = 0
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == p_pos or cur == o_pos:
                break
            c = board.get_cell(cur)
            if c == Cell.SPACE:
                space_len += 1
            else:
                break
        # A lane of k open squares could eventually yield up to CARPET_PTS[k]
        # but requires k turns of priming first → discount heavily
        if space_len >= 2:
            total += CARPET_POINTS_TABLE.get(min(space_len, 7), 0) * 0.15
    return total


def evaluate(board: Board, belief: Optional['RatBelief']) -> float:
    """
    Static evaluation from player_worker's perspective (positive = good for us).

    Components
    ----------
    score_delta   : actual pts difference
    potential     : carpet points reachable from our position minus opponent's
    rat_ev        : expected value of best rat search (using HMM belief)
    centrality    : reward being near the centre (more carpet opportunities)
    """
    pw = board.player_worker
    ow = board.opponent_worker

    score_delta = float(pw.get_points() - ow.get_points())

    my_pot  = _carpet_potential(board, pw.get_location())
    opp_pot = _carpet_potential(board, ow.get_location())
    pot_delta = my_pot - opp_pot

    rat = belief.rat_ev() if belief is not None else 0.0

    centre_bonus = float(_dist(ow.get_location(), _CENTRE) - _dist(pw.get_location(), _CENTRE))

    return (W_SCORE      * score_delta +
            W_POTENTIAL  * pot_delta   +
            W_RAT        * rat         +
            W_CENTRALITY * centre_bonus)


# ═══════════════════════════════════════════════════════════════════════════════
#  Move generation (no searches; no carpet-1)
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_moves(board: Board) -> List[Move]:
    """
    Generate all valid non-search moves, excluding carpet length 1 (= -1 pts).
    Ordered: long carpets first, then primes, then plains.
    """
    raw = board.get_valid_moves(exclude_search=True)
    result = []
    for m in raw:
        if m.move_type == MoveType.CARPET and m.roll_length == 1:
            continue   # -1 pts, never worth it unless forced
        result.append(m)

    # Order: carpet (descending length) > prime > plain
    def key(m):
        if m.move_type == MoveType.CARPET:
            return 200 + CARPET_POINTS_TABLE.get(m.roll_length, 0)
        if m.move_type == MoveType.PRIME:
            return 100
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
    Standard negamax with alpha-beta pruning and iterative deepening.

    Convention: every call returns a value from the perspective of
    the CURRENT player (player_worker on the board passed in).

    After forecast_move(m), the board's internal turn flips but
    player_worker/opponent_worker are NOT swapped.  We call
    reverse_perspective() to make player_worker = the next player,
    then recurse.  This keeps the "always from player_worker's POV"
    invariant.
    """

    def __init__(self, belief: Optional[RatBelief]):
        self.belief   = belief
        self.nodes    = 0
        self._dead    = 0.0

    # ── Public entry ────────────────────────────────────────────────────

    def best_move(self, board: Board, budget: float) -> Move:
        self._dead = time.perf_counter() + budget
        moves = _gen_moves(board)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))

        best = moves[0]
        for depth in range(1, _MAX_DEPTH + 1):
            if self._dead - time.perf_counter() < _MIN_SECS:
                break
            self.nodes = 0
            try:
                _, mv = self._negamax(board, depth, -INF, INF)
                if mv is not None:
                    best = mv
            except _Timeout:
                break   # keep best from last complete depth
        return best

    # ── Core negamax ─────────────────────────────────────────────────────

    def _negamax(self, board: Board, depth: int,
                 alpha: float, beta: float) -> Tuple[float, Optional[Move]]:

        if time.perf_counter() >= self._dead:
            raise _Timeout()

        self.nodes += 1

        if board.is_game_over() or depth == 0:
            return evaluate(board, self.belief), None

        moves = _gen_moves(board)
        if not moves:
            return evaluate(board, self.belief), None

        best_val  = -INF
        best_move = None

        for m in moves:
            child = board.forecast_move(m, check_ok=True)
            if child is None:
                continue
            # child still has player_worker = US (the mover).
            # Swap so recursive call sees it from the next player's POV.
            child.reverse_perspective()

            child_val, _ = self._negamax(child, depth - 1, -beta, -alpha)
            # child_val is from the NEXT player's POV → negate for ours
            val = -child_val

            if val > best_val:
                best_val  = val
                best_move = m

            alpha = max(alpha, val)
            if alpha >= beta:
                break   # cutoff

        return best_val, best_move


# ═══════════════════════════════════════════════════════════════════════════════
#  Player Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:
    """
    Negamax + HMM agent.

    Each turn:
    1. Update HMM with sensor data and any search results.
    2. If rat belief gives search EV above a dynamic threshold → Search.
    3. Otherwise → run negamax with iterative deepening.
    """

    def __init__(self, board_state, transition_matrix=None,
                 time_left: Callable = None):
        self.belief  = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.engine  = Negamax(self.belief)
        self.turn    = 0
        self._prev_search = (None, False)

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

        # ── Update HMM ─────────────────────────────────────────────────
        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(
                board_state.player_worker.get_location(), dist)

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
        # Budget: spread time evenly over remaining turns, cap at 5 s.
        if turns_left > 0:
            per_turn = t_remaining / turns_left
        else:
            per_turn = t_remaining
        budget = max(0.05, min(5.0, per_turn * 0.8))

        return self.engine.best_move(board, budget)
