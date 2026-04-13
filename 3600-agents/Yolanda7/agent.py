"""
Carpet-game tournament agent — v4.

Improvements over v3
====================
IMP 1 — OPPONENT-AWARE CARPET POTENTIAL:
        _carpet_potential now accepts the other player's position and
        discounts primed runs the opponent is close enough to steal.
        Runs where the opponent is 0–1 steps from the rollable end are
        heavily discounted (×0.15); 2–3 steps moderately (×0.55).

IMP 2 — SEARCH MOVES COMPETE WITH BOARD MOVES:
        The hardcoded patience threshold is replaced by a two-stage
        decision: (a) a high-confidence fast path (EV >= 1.5) that skips
        negamax entirely, and (b) a post-negamax comparison that only
        searches when the opportunity cost of skipping a board move is
        lower than the expected rat points.  The agent never sacrifices
        a valuable carpet roll for a marginal rat search.

IMP 3 — PRINCIPAL VARIATION SEARCH (PVS):
        After the first child (expected to be best thanks to move
        ordering), remaining children are searched with a zero-width
        scout window.  Re-search at full width only if the scout
        indicates the move might be better than the current best.

IMP 4 — ASPIRATION WINDOWS:
        Iterative deepening uses a narrow (+-1.5) window around the
        previous depth's value from depth >= 3 onward.  Falls back to
        a full-width re-search on fail-high / fail-low.

IMP 5 — BETTER TIME MANAGEMENT:
        Early game (turns 1-13): 1.5x time allocation for strategic
        foundation.  Mid game (14-25): 1.0x.  Late game (26-40): 0.6x
        because positions are simpler and branching factor is lower.

IMP 6 — RUN EXTENSION BONUS IN MOVE ORDERING:
        Prime steps whose departure square (player_pos) is adjacent to
        existing primed cells get a bonus proportional to the number of
        primed neighbours.  This tends to extend runs rather than start
        isolated new ones, improving the quality of alpha-beta pruning.

IMP 7 — OPPONENT MISS NEIGHBOUR BOOST:
        When the opponent searches a cell and misses, the four cardinal
        neighbours receive a mild 1.15x belief boost.  The opponent's
        decision to search there is weak evidence the rat is nearby.
        Conservative multiplier avoids corrupting our own observations.

Retained from v3
================
- Ply-indexed transposition table (cleared each turn)
- Pre-predicted belief snapshots (O(max_depth) instead of O(b^d))
- T^1000 spawn prior on rat capture
- Dual-pass carpet potential
- Vectorised HMM
- Killer moves + history heuristic
- Dynamic W_RAT
"""

from collections.abc import Callable
from typing import Dict, List, Optional, Tuple
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
DIST_ERR_P   = (0.12, 0.70, 0.12, 0.06)
DIST_ERR_OFF = (-1,    0,    1,    2)

# Search budget
_TIME_BUFFER = 0.25
_MAX_DEPTH   = 8
_MIN_SECS    = 0.03

# Heuristic weights
W_SCORE      = 1.0
W_POTENTIAL  = 0.45
W_RAT        = 0.60
W_CENTRALITY = 0.10

_MAX_DIST = (BOARD_SIZE - 1) * 2
_MAX_OBS  = _MAX_DIST + max(DIST_ERR_OFF) + 1

# IMP 4 — aspiration window half-width
_ASP_WINDOW = 1.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]

def _loc(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)

def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

_CENTRE = (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1)


def _board_key(board: Board, ply: int) -> tuple:
    """Hashable TT key.  Includes ply so different belief ages never collide."""
    pw = board.player_worker
    ow = board.opponent_worker
    try:
        return (
            ply,
            int(board.primed),
            int(board.carpeted),
            pw.get_location(),
            ow.get_location(),
            pw.get_points(),
            ow.get_points(),
            pw.turns_left,
        )
    except AttributeError:
        return (
            ply,
            hash(board),
            pw.get_location(),
            ow.get_location(),
            pw.get_points(),
            ow.get_points(),
            pw.turns_left,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Hidden Markov Model
# ═══════════════════════════════════════════════════════════════════════════════

class RatBelief:
    """Probability distribution over all 64 cells for the rat's position."""

    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.b = np.full(N, 1.0 / N)

        locs = [_loc(i) for i in range(N)]
        self.dist_matrix = np.array(
            [[abs(locs[i][0] - locs[j][0]) + abs(locs[i][1] - locs[j][1])
              for j in range(N)]
             for i in range(N)],
            dtype=np.int32,
        )

        self.like_table = np.zeros((_MAX_DIST + 1, _MAX_OBS + 1), dtype=np.float64)
        for actual in range(_MAX_DIST + 1):
            for off, prob in zip(DIST_ERR_OFF, DIST_ERR_P):
                obs = max(0, actual + off)
                if obs <= _MAX_OBS:
                    self.like_table[actual, obs] += prob

        # T^1000 spawn prior
        e0 = np.zeros(N, dtype=np.float64)
        e0[_idx((0, 0))] = 1.0
        try:
            T_1000 = np.linalg.matrix_power(self.T, 1000)
            self.spawn_prior = e0 @ T_1000
        except np.linalg.LinAlgError:
            mat = self.T.copy()
            result = e0.copy()
            remaining = 1000
            while remaining > 0:
                if remaining % 2 == 1:
                    result = result @ mat
                mat = mat @ mat
                remaining //= 2
            self.spawn_prior = result
        sp_sum = self.spawn_prior.sum()
        if sp_sum > 0:
            self.spawn_prior /= sp_sum

        self.b = self.spawn_prior.copy()

    # ── Belief updates ────────────────────────────────────────────────────

    def predict(self):
        self.b = self.b @ self.T
        s = self.b.sum()
        if s > 0:
            self.b /= s

    def update_noise(self, board: Board, noise_type):
        ni = int(noise_type)
        likelihood = np.fromiter(
            (NOISE_PROBS.get(board.get_cell(_loc(i)),
                             NOISE_PROBS[Cell.SPACE])[ni]
             for i in range(N)),
            dtype=np.float64, count=N,
        )
        self.b *= likelihood
        self._norm()

    def update_distance(self, pos: Tuple[int, int], observed: int):
        p_idx = _idx(pos)
        actuals = self.dist_matrix[p_idx]
        obs_clamped = min(observed, _MAX_OBS)
        likelihoods = self.like_table[actuals, obs_clamped]
        self.b *= likelihoods
        self._norm()

    def update_search(self, loc: Tuple[int, int], found: bool,
                      is_opponent: bool = False):
        """
        IMP 7: on opponent miss, mildly boost the four cardinal neighbours
        of the searched cell — the opponent's HMM thought the rat was there,
        so it is weak evidence the rat is nearby.
        """
        if found:
            self.b = self.spawn_prior.copy()
        else:
            self.b[_idx(loc)] = 0.0
            if is_opponent:
                x, y = loc
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        self.b[_idx((nx, ny))] *= 1.15
            self._norm()

    def _norm(self):
        s = self.b.sum()
        if s > 1e-15:
            self.b /= s
        else:
            self.b = self.spawn_prior.copy()

    # ── Tree-search helpers ───────────────────────────────────────────────

    def branch_copy(self) -> 'RatBelief':
        other = object.__new__(RatBelief)
        other.T           = self.T
        other.dist_matrix = self.dist_matrix
        other.like_table  = self.like_table
        other.spawn_prior = self.spawn_prior
        other.b           = self.b.copy()
        return other

    # ── Queries ──────────────────────────────────────────────────────────

    def best_search_ev(self) -> Tuple[Tuple[int, int], float]:
        i  = int(np.argmax(self.b))
        p  = float(self.b[i])
        ev = p * RAT_BONUS - (1.0 - p) * RAT_PENALTY
        return _loc(i), ev

    def rat_ev(self) -> float:
        _, ev = self.best_search_ev()
        return ev

    def top_n(self, n: int = 3):
        idx = np.argsort(self.b)[-n:][::-1]
        return [(_loc(int(i)), float(self.b[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def _carpet_potential(board: Board, pos: Tuple[int, int],
                      other_pos: Tuple[int, int]) -> float:
    """
    IMP 1: opponent-aware carpet potential.

    For each direction, count the primed run and the open-space lane from
    ``pos``.  Primed runs are discounted when ``other_pos`` (the opponent)
    is close enough to roll them first.
    """
    p_pos = board.player_worker.get_location()
    o_pos = board.opponent_worker.get_location()
    total = 0.0

    for d in ALL_DIRS:
        # ── Pass 1: primed run ────────────────────────────────────────
        primed_run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == p_pos or cur == o_pos:
                break
            if board.get_cell(cur) == Cell.PRIMED:
                primed_run += 1
            else:
                break

        if primed_run >= 2:
            base_pts = CARPET_POINTS_TABLE.get(primed_run, 0)
            # IMP 1: discount if opponent can reach the roll position
            opp_steps = _dist(other_pos, pos)
            if opp_steps <= 1:
                base_pts *= 0.15
            elif opp_steps <= 3:
                base_pts *= 0.55
            total += base_pts

        # ── Pass 2: open space lane ───────────────────────────────────
        space_run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == p_pos or cur == o_pos:
                break
            if board.get_cell(cur) == Cell.SPACE:
                space_run += 1
            else:
                break
        if space_run >= 2:
            total += CARPET_POINTS_TABLE.get(min(space_run, 7), 0) * 0.15

    return total


def _evaluate_board(board: Board) -> float:
    """Belief-independent evaluation (safe to cache in TT)."""
    pw = board.player_worker
    ow = board.opponent_worker
    my_pos  = pw.get_location()
    opp_pos = ow.get_location()

    score_delta = float(pw.get_points() - ow.get_points())

    my_pot  = _carpet_potential(board, my_pos,  opp_pos)
    opp_pot = _carpet_potential(board, opp_pos, my_pos)
    pot_delta = my_pot - opp_pot

    centre_bonus = float(
        _dist(opp_pos, _CENTRE) - _dist(my_pos, _CENTRE)
    )

    return (W_SCORE      * score_delta +
            W_POTENTIAL  * pot_delta   +
            W_CENTRALITY * centre_bonus)


def _evaluate_rat(board: Board, belief: Optional[RatBelief]) -> float:
    """Belief-dependent rat term (NOT cached in TT)."""
    if belief is None:
        return 0.0
    turns_left = board.player_worker.turns_left
    frac_done  = 1.0 - turns_left / max(1, MAX_TURNS_PER_PLAYER)
    rat_weight = W_RAT * (1.0 + frac_done * 0.5)
    return rat_weight * belief.rat_ev()


def evaluate(board: Board, belief: Optional[RatBelief]) -> float:
    return _evaluate_board(board) + _evaluate_rat(board, belief)


# ═══════════════════════════════════════════════════════════════════════════════
#  Move helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _move_key(m: Optional[Move]) -> Optional[tuple]:
    if m is None:
        return None
    return (
        m.move_type,
        getattr(m, 'direction', None),
        getattr(m, 'roll_length', 0),
    )


def _gen_moves(board: Board) -> List[Move]:
    """Valid non-search moves, excluding carpet-1 (always -1 pts)."""
    raw = board.get_valid_moves(exclude_search=True)
    result = []
    for m in raw:
        if m.move_type == MoveType.CARPET and m.roll_length == 1:
            continue
        result.append(m)
    return result


def _order_moves(moves: List[Move],
                 depth: int,
                 killers: List[List[Optional[tuple]]],
                 history: Dict[tuple, int],
                 board: Board,
                 player_pos: Tuple[int, int]) -> None:
    """
    IMP 6: move ordering with run-extension awareness.

    Prime steps whose departure square (player_pos) is adjacent to
    existing primed cells get a bonus proportional to the number of
    primed neighbours.  This tends to extend runs rather than start
    isolated new ones, improving the quality of alpha-beta pruning.
    """
    k0 = killers[depth][0]
    k1 = killers[depth][1]
    max_hist = max(history.values()) if history else 1

    # IMP 6: precompute primed-neighbour count of player_pos once
    primed_neighbours = 0
    for check_d in ALL_DIRS:
        nb = loc_after_direction(player_pos, check_d)
        if board.is_valid_cell(nb) and board.get_cell(nb) == Cell.PRIMED:
            primed_neighbours += 1

    def sort_key(m: Move) -> float:
        mk = _move_key(m)

        if m.move_type == MoveType.CARPET:
            return 600.0 + CARPET_POINTS_TABLE.get(m.roll_length, 0)

        if mk == k0:
            return 500.0
        if mk == k1:
            return 490.0

        hist = history.get(mk, 0)
        hist_norm = 89.0 * hist / max_hist if max_hist > 0 else 0.0

        if m.move_type == MoveType.PRIME:
            ext_bonus = primed_neighbours * 6.0
            return 50.0 + ext_bonus + hist_norm

        return hist_norm

    moves.sort(key=sort_key, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Negamax with PVS + aspiration windows
# ═══════════════════════════════════════════════════════════════════════════════

INF = float('inf')
_EXACT = 0
_LOWER = 1
_UPPER = 2


class _Timeout(Exception):
    pass


class Negamax:
    """
    Negamax with alpha-beta, iterative deepening, PVS, aspiration windows,
    transposition table, killer moves, and history heuristic.
    """

    def __init__(self, belief: Optional[RatBelief]):
        self.belief  = belief
        self.nodes   = 0
        self._dead   = 0.0
        self._tt: Dict[tuple, Tuple[int, float, int]] = {}
        self._beliefs_by_ply: List[Optional[RatBelief]] = []
        self._current_max_depth = 0

    # ── Public entry ────────────────────────────────────────────────────

    def best_move(self, board: Board, budget: float) -> Move:
        self._dead = time.perf_counter() + budget

        moves = _gen_moves(board)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))

        self._tt.clear()

        self._killers: List[List[Optional[tuple]]] = [
            [None, None] for _ in range(_MAX_DEPTH + 2)
        ]
        self._history: Dict[tuple, int] = {}

        # Pre-predict belief snapshots
        if self.belief is not None:
            snapshots: List[Optional[RatBelief]] = [self.belief.branch_copy()]
            for _ in range(1, _MAX_DEPTH + 1):
                prev = snapshots[-1].branch_copy()
                prev.predict()
                snapshots.append(prev)
            self._beliefs_by_ply = snapshots
        else:
            self._beliefs_by_ply = [None] * (_MAX_DEPTH + 1)

        best = moves[0]
        prev_val = 0.0

        for depth in range(1, _MAX_DEPTH + 1):
            if self._dead - time.perf_counter() < _MIN_SECS:
                break
            self.nodes = 0
            self._current_max_depth = depth

            try:
                # IMP 4: aspiration window from depth >= 3
                if depth >= 3 and abs(prev_val) < 100.0:
                    lo = prev_val - _ASP_WINDOW
                    hi = prev_val + _ASP_WINDOW
                    val, mv = self._negamax(board, depth, lo, hi)

                    if val <= lo or val >= hi:
                        # Failed outside window — full re-search
                        val, mv = self._negamax(board, depth, -INF, INF)
                else:
                    val, mv = self._negamax(board, depth, -INF, INF)

                if mv is not None:
                    best = mv
                    prev_val = val
            except _Timeout:
                break

        return best

    # ── Core negamax with PVS ────────────────────────────────────────────

    def _negamax(self, board: Board, depth: int,
                 alpha: float, beta: float) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= self._dead:
            raise _Timeout()

        self.nodes += 1

        ply = self._current_max_depth - depth
        ply_clamped = min(ply, len(self._beliefs_by_ply) - 1)
        ply_belief = self._beliefs_by_ply[ply_clamped]

        # ── TT probe ─────────────────────────────────────────────────
        key   = _board_key(board, ply)
        entry = self._tt.get(key)
        if entry is not None:
            tt_depth, tt_val, tt_flag = entry
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    return tt_val, None
                if tt_flag == _LOWER:
                    alpha = max(alpha, tt_val)
                if tt_flag == _UPPER:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val, None

        # ── Terminal / leaf ───────────────────────────────────────────
        if board.is_game_over() or depth == 0:
            val = evaluate(board, ply_belief)
            self._tt[key] = (depth, val, _EXACT)
            return val, None

        moves = _gen_moves(board)
        if not moves:
            val = evaluate(board, ply_belief)
            self._tt[key] = (depth, val, _EXACT)
            return val, None

        # ── Move ordering (IMP 6: run-extension awareness) ───────────
        player_pos = board.player_worker.get_location()
        _order_moves(moves, depth, self._killers, self._history,
                     board, player_pos)

        best_val  = -INF
        best_move = None
        orig_alpha = alpha
        first_valid = True

        for m in moves:
            child = board.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()

            # ── IMP 3: PVS ───────────────────────────────────────────
            if first_valid:
                child_val, _ = self._negamax(child, depth - 1,
                                             -beta, -alpha)
                first_valid = False
            else:
                # Zero-width scout
                child_val, _ = self._negamax(child, depth - 1,
                                             -alpha - 1, -alpha)
                if -child_val > alpha and -child_val < beta:
                    # Scout failed — re-search at full width
                    child_val, _ = self._negamax(child, depth - 1,
                                                 -beta, -alpha)

            val = -child_val

            if val > best_val:
                best_val  = val
                best_move = m

            alpha = max(alpha, val)
            if alpha >= beta:
                mk = _move_key(m)
                if mk is not None:
                    if mk != self._killers[depth][0]:
                        self._killers[depth][1] = self._killers[depth][0]
                        self._killers[depth][0] = mk
                    self._history[mk] = (self._history.get(mk, 0) +
                                         depth * depth)
                break

        # ── TT store ─────────────────────────────────────────────────
        if best_val <= orig_alpha:
            flag = _UPPER
        elif best_val >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        existing = self._tt.get(key)
        if existing is None or depth >= existing[0]:
            self._tt[key] = (depth, best_val, flag)

        return best_val, best_move


# ═══════════════════════════════════════════════════════════════════════════════
#  Player Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:

    def __init__(self, board_state, transition_matrix=None,
                 time_left: Callable = None):
        self.belief       = (RatBelief(transition_matrix)
                             if transition_matrix is not None else None)
        self.engine       = Negamax(self.belief)
        self.turn         = 0
        self._prev_search = (None, False)

    # ── Commentate ───────────────────────────────────────────────────────

    def commentate(self) -> str:
        if self.belief is None:
            return "No belief."
        top   = self.belief.top_n(3)
        parts = [f"{loc}:{p:.3f}" for loc, p in top]
        return (f"Turn {self.turn} | rat top: {', '.join(parts)} "
                f"| nodes: {self.engine.nodes} | tt: {len(self.engine._tt)}")

    # ── Play ─────────────────────────────────────────────────────────────

    def play(self, board_state: Board, sensor_data: Tuple,
             time_left: Callable) -> Move:
        self.turn += 1
        noise, dist = sensor_data

        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(
                board_state.player_worker.get_location(), dist)

            # Opponent's last search — IMP 7: pass is_opponent=True
            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc is not None:
                self.belief.update_search(opp_loc, opp_hit,
                                          is_opponent=True)

            # Our own last search
            my_loc, my_hit = board_state.player_search
            if my_loc is not None and my_loc != self._prev_search[0]:
                self.belief.update_search(my_loc, my_hit,
                                          is_opponent=False)
                self._prev_search = (my_loc, my_hit)

        return self._decide(board_state, time_left)

    # ── Decision logic ───────────────────────────────────────────────────

    def _decide(self, board: Board, time_left: Callable) -> Move:
        turns_left  = board.player_worker.turns_left
        t_remaining = time_left() - _TIME_BUFFER

        if t_remaining <= 0:
            moves = _gen_moves(board)
            return moves[0] if moves else Move.search((0, 0))

        # ── Pre-compute search info ──────────────────────────────────
        search_loc = None
        search_ev  = -INF
        if self.belief is not None:
            search_loc, search_ev = self.belief.best_search_ev()

        # ── IMP 2a: high-confidence fast path (skip negamax) ─────────
        if search_ev >= 1.5:
            return Move.search(search_loc)

        # ── IMP 5: time allocation weighted by game phase ────────────
        if turns_left > 27:
            time_frac = 1.5
        elif turns_left > 15:
            time_frac = 1.0
        else:
            time_frac = 0.6
        per_turn = t_remaining / max(1, turns_left) * time_frac
        budget   = max(0.05, min(5.0, per_turn * 0.85))

        # ── Run negamax ──────────────────────────────────────────────
        best_move = self.engine.best_move(board, budget)

        # ── IMP 2b: post-negamax search comparison ───────────────────
        if search_ev > 0 and search_loc is not None:
            # Don't sacrifice a high-value carpet roll
            if (best_move.move_type == MoveType.CARPET and
                    best_move.roll_length >= 2):
                carpet_pts = CARPET_POINTS_TABLE.get(
                    best_move.roll_length, 0)
                if search_ev > carpet_pts:
                    return Move.search(search_loc)
            else:
                # Board move is plain/prime — low opportunity cost.
                # Late game: lower threshold (rat hunting is relatively
                # more valuable as carpet opportunities shrink).
                frac_done = 1.0 - turns_left / max(1, MAX_TURNS_PER_PLAYER)
                threshold = max(0.0, 0.4 * (1.0 - frac_done))
                if search_ev >= threshold:
                    return Move.search(search_loc)

        return best_move