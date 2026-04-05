"""
Competitive carpet-game agent.

Architecture
============
1.  RatBelief  – Hidden Markov Model over all 64 cells.
2.  Heuristic  – evaluates a board position from the player's perspective,
                 accounting for:
                 * score delta (our points − opponent points)
                 * carpet potential reachable from current position
                 * rat search expected value (best cell in HMM × RAT_BONUS)
                 * turns / time remaining pressure
3.  Expectiminimax with iterative deepening + alpha-beta pruning.
    The rat is modelled via the HMM belief: its contribution enters through
    the heuristic's rat-EV term rather than explicit chance-node branching.
4.  Move ordering – carpet > prime > plain so alpha-beta cuts happen early.
"""

from collections.abc import Callable
from typing import List, Optional, Tuple
import random
import time

from game import board as board_module, move as move_module, enums
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

# P(noise_type | cell_type)  — indices: 0=squeak, 1=scratch, 2=squeal
NOISE_PROBS = {
    Cell.BLOCKED: (0.5,  0.3,  0.2),
    Cell.SPACE:   (0.7,  0.15, 0.15),
    Cell.PRIMED:  (0.1,  0.8,  0.1),
    Cell.CARPET:  (0.1,  0.1,  0.8),
}

# P(observed_dist = actual + offset)
DIST_ERROR_PROBS   = (0.12, 0.70, 0.12, 0.06)
DIST_ERROR_OFFSETS = (-1,   0,    1,    2)

# Seconds to keep in reserve so we never time-out
_TIME_BUFFER = 0.20

# Iterative deepening ceiling and minimum time to start a new depth
_MAX_DEPTH   = 6
_MIN_SECONDS = 0.05

# Heuristic weights
_W_SCORE_DELTA  = 1.0   # raw point difference (most important)
_W_POTENTIAL    = 0.40  # carpet potential differential
_W_RAT_EV       = 0.55  # expected value of best rat search
_W_MOBILITY     = 0.04  # number of moves available
_W_TURNS_REMAIN = 0.03  # turns remaining edge


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _loc_to_idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _idx_to_loc(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ═══════════════════════════════════════════════════════════════════════════════
#  Hidden Markov Model – Rat Belief
# ═══════════════════════════════════════════════════════════════════════════════

class RatBelief:
    """
    Maintains P(rat at cell i) for all 64 cells.

    Call each turn in order:
        belief.predict()               # rat moves one step via T
        belief.update_noise(b, n)      # condition on noise heard
        belief.update_distance(pos, d) # condition on distance estimate
        belief.update_search(loc, hit) # for any search that happened
    """

    def __init__(self, T):
        import numpy as np
        try:
            self.T = np.array(T, dtype=np.float64)
        except Exception:
            self.T = T
        self.belief = [1.0 / N] * N

    # ── HMM operations ────────────────────────────────────────────────

    def predict(self):
        import numpy as np
        b = np.array(self.belief, dtype=np.float64)
        b = b @ self.T
        s = b.sum()
        self.belief = (b / s).tolist() if s > 0 else [1.0 / N] * N

    def update_noise(self, board_state: Board, noise_type):
        noise_idx = int(noise_type)
        for idx in range(N):
            try:
                cell = board_state.get_cell(_idx_to_loc(idx))
            except Exception:
                cell = Cell.SPACE
            self.belief[idx] *= NOISE_PROBS.get(cell, NOISE_PROBS[Cell.SPACE])[noise_idx]
        self._normalise()

    def update_distance(self, worker_pos: Tuple[int, int], observed_dist: int):
        for idx in range(N):
            actual = _manhattan(worker_pos, _idx_to_loc(idx))
            prob = 0.0
            for offset, p in zip(DIST_ERROR_OFFSETS, DIST_ERROR_PROBS):
                expected = max(0, actual + offset)
                if expected == observed_dist:
                    prob += p
            self.belief[idx] *= prob
        self._normalise()

    def update_search(self, search_loc: Tuple[int, int], found: bool):
        if found:
            self.belief = [1.0 / N] * N   # new rat → uniform prior
        else:
            self.belief[_loc_to_idx(search_loc)] = 0.0
            self._normalise()

    def _normalise(self):
        s = sum(self.belief)
        if s > 1e-15:
            inv = 1.0 / s
            self.belief = [x * inv for x in self.belief]
        else:
            self.belief = [1.0 / N] * N

    # ── Queries ───────────────────────────────────────────────────────

    def best_search_ev(self) -> Tuple[Tuple[int, int], float]:
        """EV(p) = p*RAT_BONUS − (1−p)*RAT_PENALTY = 6p − 2."""
        best_p   = max(self.belief)
        best_idx = self.belief.index(best_p)
        ev = best_p * RAT_BONUS - (1.0 - best_p) * RAT_PENALTY
        return _idx_to_loc(best_idx), ev

    def expected_search_ev(self) -> float:
        _, ev = self.best_search_ev()
        return ev

    def top_n(self, n: int = 5):
        pairs = sorted(enumerate(self.belief), key=lambda x: -x[1])[:n]
        return [(_idx_to_loc(i), p) for i, p in pairs]


# ═══════════════════════════════════════════════════════════════════════════════
#  Carpet Potential Heuristic
# ═══════════════════════════════════════════════════════════════════════════════

def _carpet_potential(board_state: Board, pos: Tuple[int, int]) -> float:
    """
    Estimate the carpet point value reachable from `pos`.

    For each of the 4 directions:
    - Count the consecutive PRIMED cells ahead (= possible carpet length).
    - Look up the point value from CARPET_POINTS_TABLE.
    - Add discounted credit for SPACE cells that could be primed in future turns.
    - Subtract distance cost to reach the start of the primed run.

    Returns a float representing expected future point value from pos.
    """
    total = 0.0

    player_bit = board_state._loc_to_bit_index(board_state.player_worker.get_location())
    opp_bit    = board_state._loc_to_bit_index(board_state.opponent_worker.get_location())

    for direction in Direction:
        primed_run = 0
        space_run  = 0
        check = pos

        for step in range(1, BOARD_SIZE):
            check = loc_after_direction(check, direction)
            if not board_state.is_valid_cell(check):
                break

            bit = board_state._loc_to_bit_index(check)
            if bit == player_bit or bit == opp_bit:
                break   # workers block carpet

            cell = board_state.get_cell(check)
            if cell == Cell.PRIMED:
                primed_run += 1
            elif cell == Cell.SPACE:
                space_run += 1
                break   # can't carpet through space
            else:
                break   # BLOCKED or CARPET stops run

        if primed_run >= 1:
            pts = CARPET_POINTS_TABLE.get(primed_run, 0)
            total += max(0.0, float(pts))

        # Potential future priming in open space
        total += space_run * 0.25

    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  Static Evaluation (Heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(board_state: Board, belief: Optional[RatBelief]) -> float:
    """
    Evaluate from the perspective of player_worker (higher = better for player).

    Components:
    1. Score delta (player_pts − opponent_pts)
    2. Carpet potential differential (my reachable carpet pts − opponent's)
    3. Rat search expected value
    4. Mobility
    5. Turns remaining edge
    """
    pw = board_state.player_worker
    ow = board_state.opponent_worker

    score_delta = float(pw.get_points() - ow.get_points())

    my_pot  = _carpet_potential(board_state, pw.get_location())
    opp_pot = _carpet_potential(board_state, ow.get_location())
    potential_delta = my_pot - opp_pot

    rat_ev = belief.expected_search_ev() if belief is not None else 0.0

    mobility = float(len(board_state.get_valid_moves(exclude_search=True)))

    turns_edge = float(pw.turns_left - ow.turns_left)

    return (
        _W_SCORE_DELTA  * score_delta    +
        _W_POTENTIAL    * potential_delta +
        _W_RAT_EV       * rat_ev          +
        _W_MOBILITY     * mobility        +
        _W_TURNS_REMAIN * turns_edge
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Move Ordering
# ═══════════════════════════════════════════════════════════════════════════════

def _move_order_key(m: Move) -> float:
    """Higher priority = try first (maximises alpha-beta cutoffs)."""
    if m.move_type == MoveType.CARPET:
        return 100.0 + CARPET_POINTS_TABLE.get(m.roll_length, 0)
    if m.move_type == MoveType.PRIME:
        return 10.0
    return 1.0   # PLAIN


def _ordered_moves(board_state: Board) -> List[Move]:
    moves = board_state.get_valid_moves(exclude_search=True)
    moves.sort(key=_move_order_key, reverse=True)
    return moves


# ═══════════════════════════════════════════════════════════════════════════════
#  Expectiminimax with Alpha-Beta + Iterative Deepening
# ═══════════════════════════════════════════════════════════════════════════════

INF = float('inf')


class _TimeUp(Exception):
    pass


class Searcher:
    """
    Iterative-deepening alpha-beta expectiminimax search.

    The game alternates MAX (our turn) and MIN (opponent's turn) nodes.
    forecast_move() flips is_player_a_turn internally, so after calling it
    we reverse_perspective() so that player_worker always refers to
    "whoever is to move next" — this keeps evaluate() symmetric.
    """

    def __init__(self, belief: Optional[RatBelief]):
        self.belief   = belief
        self.nodes     = 0
        self._deadline = 0.0

    def search(self, root: Board, time_budget: float) -> Move:
        """Run iterative deepening; return best move found within budget."""
        self._deadline = time.perf_counter() + time_budget

        # Depth-0 fallback: best greedy move
        moves = _ordered_moves(root)
        if not moves:
            loc, _ = self.belief.best_search_ev() if self.belief else ((0, 0), -INF)
            return Move.search(loc)

        best_move = moves[0]

        for depth in range(1, _MAX_DEPTH + 1):
            remaining = self._deadline - time.perf_counter()
            if remaining < _MIN_SECONDS:
                break

            self.nodes = 0
            try:
                val, mv = self._maxi(root, depth, -INF, INF)
                if mv is not None:
                    best_move = mv
            except _TimeUp:
                break

        return best_move

    # ── Alpha-Beta tree ────────────────────────────────────────────────

    def _maxi(self, b: Board, depth: int, alpha: float, beta: float):
        """MAX node — our move."""
        if time.perf_counter() >= self._deadline:
            raise _TimeUp()

        self.nodes += 1

        if b.is_game_over() or depth == 0:
            return evaluate(b, self.belief), None

        best_val  = -INF
        best_move = None

        for m in _ordered_moves(b):
            child = b.forecast_move(m, check_ok=False)
            if child is None:
                continue
            # forecast_move ends the turn; now it's the opponent's turn.
            # Swap perspective so player_worker = opponent, then call _mini.
            child.reverse_perspective()
            val, _ = self._mini(child, depth - 1, alpha, beta)
            # _mini returns value from opponent's perspective; negate for ours.
            val = -val

            if val > best_val:
                best_val  = val
                best_move = m

            alpha = max(alpha, best_val)
            if alpha >= beta:
                break   # β-cutoff

        return best_val, best_move

    def _mini(self, b: Board, depth: int, alpha: float, beta: float):
        """MIN node — opponent's move (board is already from their perspective)."""
        if time.perf_counter() >= self._deadline:
            raise _TimeUp()

        self.nodes += 1

        if b.is_game_over() or depth == 0:
            # evaluate is always from player_worker's POV.
            # Since we reversed perspective, player_worker is the opponent.
            # Negate so the value is from the MAX player's POV.
            return -evaluate(b, self.belief), None

        best_val  = INF
        best_move = None

        for m in _ordered_moves(b):
            child = b.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val, _ = self._maxi(child, depth - 1, alpha, beta)
            val = -val   # flip back to MAX player's POV

            if val < best_val:
                best_val  = val
                best_move = m

            beta = min(beta, best_val)
            if alpha >= beta:
                break   # α-cutoff

        return best_val, best_move


# ═══════════════════════════════════════════════════════════════════════════════
#  Player Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:
    """
    Expectiminimax agent with HMM rat tracking and an advanced heuristic.

    Strategy
    --------
    Each turn:
    1.  Update HMM with this turn's sensor data and any search results.
    2.  If rat belief is concentrated enough (EV above a dynamic threshold),
        fire a Search move immediately — rat points compound quickly.
    3.  Otherwise run expectiminimax within the per-turn time budget to find
        the best carpeting / priming sequence.
    """

    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self.belief   = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.searcher = Searcher(self.belief)
        self.turn_num = 0
        self._prev_player_search = (None, False)
        self._last_nodes = 0

    # ------------------------------------------------------------------
    def commentate(self) -> str:
        if self.belief is None:
            return "No belief model."
        top = self.belief.top_n(3)
        parts = [f"{loc}:{p:.3f}" for loc, p in top]
        return (
            f"Turn {self.turn_num} | "
            f"Top rat: {', '.join(parts)} | "
            f"Nodes searched last turn: {self._last_nodes}"
        )

    # ------------------------------------------------------------------
    def play(self, board_state: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turn_num += 1
        noise, dist = sensor_data

        # ── Update HMM ────────────────────────────────────────────────
        if self.belief is not None:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(board_state.player_worker.get_location(), dist)

            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc is not None:
                self.belief.update_search(opp_loc, opp_hit)

            my_loc, my_hit = board_state.player_search
            if my_loc is not None and my_loc != self._prev_player_search[0]:
                self.belief.update_search(my_loc, my_hit)
                self._prev_player_search = (my_loc, my_hit)

        return self._decide(board_state, time_left)

    # ------------------------------------------------------------------
    def _decide(self, board_state: Board, time_left: Callable) -> Move:
        turns_left = board_state.player_worker.turns_left
        t_left     = time_left() - _TIME_BUFFER

        if t_left <= 0:
            # Emergency: any valid move
            moves = board_state.get_valid_moves(exclude_search=True)
            return moves[0] if moves else Move.search((0, 0))

        # ── Rat search decision ────────────────────────────────────────
        # EV = 6p − 2; break-even p = 1/3 ≈ 0.333
        # We add a dynamic extra threshold:
        #   - Early game (many turns left): require higher confidence to not
        #     waste a turn on a low-EV search.
        #   - Late game: lower threshold; points matter more at the end.
        if self.belief is not None:
            search_loc, search_ev = self.belief.best_search_ev()
            turns_played = MAX_TURNS_PER_PLAYER - turns_left
            # Threshold decreases from 2.0 → 0.0 over the first 30 turns played
            threshold = max(0.0, 2.0 - turns_played * (2.0 / 30.0))
            if search_ev >= threshold:
                return Move.search(search_loc)

        # ── Expectiminimax ─────────────────────────────────────────────
        # Allocate time budget: try to spread time evenly across remaining turns
        # but never use more than 5 s on one move.
        if turns_left > 0:
            per_turn = t_left / turns_left
        else:
            per_turn = t_left
        budget = max(0.05, min(5.0, per_turn * 0.75))

        best_move = self.searcher.search(board_state, budget)
        self._last_nodes = self.searcher.nodes
        return best_move
