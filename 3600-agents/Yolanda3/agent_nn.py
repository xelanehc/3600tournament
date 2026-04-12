"""
agent_nn.py — Neural-network guided negamax agent.

This agent uses a trained Flax neural network for board evaluation and move ordering,
falling back to hand-crafted heuristics if no weights are available. It performs
negamax search with iterative deepening and alpha-beta pruning.

Changes from baseline negamax:
- Uses NN value head for leaf evaluation instead of heuristic
- Uses NN policy head for move ordering instead of carpet > prime > plain
- No RatBelief/HMM; rat searches use simple EV threshold (uniform prior)
- Focuses on carpeting strategy, learns search policy from self-play data

Weight loading: Loads from models/weights.npz at init, or uses injected _EVAL_PARAMS
for evaluation runs to avoid disk I/O.
"""

from collections.abc import Callable
from typing import List, Optional, Tuple
import os, time
import numpy as np

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE,
    loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import net as NET

# Path to trained weights (relative to this file's directory)
NET_WEIGHTS_PATH = os.path.join(_HERE, 'models', 'weights.npz')

# Injection point for train.py evaluation (avoids disk I/O per game)
_EVAL_PARAMS = None   # set to a Flax params pytree by train.py


# ── Constants ─────────────────────────────────────────────────────────────────
N        = BOARD_SIZE * BOARD_SIZE
ALL_DIRS = list(Direction)

_TIME_BUFFER = 0.25  # Time buffer before timeout
_MAX_DEPTH   = 6     # Max search depth
_MIN_SECS    = 0.03  # Min seconds per depth

# Heuristic fallback weights (no rat term — no HMM here)
W_SCORE      = 1.0   # Weight for score difference
W_POTENTIAL  = 0.45  # Weight for carpet potential
W_CENTRALITY = 0.10  # Weight for centrality

INF     = float('inf')
_CENTRE = (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1)  # Board center

def _dist(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan distance


# ── Hand-crafted heuristic (fallback, no rat term) ────────────────────────────

def _primed_run(board, start, direction, p_pos, o_pos):
    # Count consecutive primed cells in a direction from start, stopping at workers or board edge
    length = 0; cur = start
    for _ in range(BOARD_SIZE - 1):
        cur = loc_after_direction(cur, direction)
        if not board.is_valid_cell(cur): break
        if cur in (p_pos, o_pos): break
        if board.get_cell(cur) == Cell.PRIMED: length += 1
        else: break
    return length

def _carpet_potential(board, pos):
    # Estimate carpet scoring potential from a position by checking runs in all directions
    p_pos = board.player_worker.get_location()
    o_pos = board.opponent_worker.get_location()
    total = 0.0
    for d in ALL_DIRS:
        run = _primed_run(board, pos, d, p_pos, o_pos)
        if run >= 2: total += CARPET_POINTS_TABLE.get(run, 0)  # Points for primed runs
        cur = pos; sl = 0
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur): break
            if cur in (p_pos, o_pos): break
            if board.get_cell(cur) == Cell.SPACE: sl += 1  # Count empty spaces
            else: break
        if sl >= 2: total += CARPET_POINTS_TABLE.get(min(sl, 7), 0) * 0.15  # Potential for future carpets
    return total

def _heuristic(board) -> float:
    """Hand-crafted board evaluation without any rat term."""
    pw = board.player_worker
    ow = board.opponent_worker
    score   = float(pw.get_points() - ow.get_points())  # Score difference
    pot     = _carpet_potential(board, pw.get_location()) - _carpet_potential(board, ow.get_location())  # Potential difference
    centre  = float(_dist(ow.get_location(), _CENTRE) - _dist(pw.get_location(), _CENTRE))  # Centrality advantage
    return W_SCORE * score + W_POTENTIAL * pot + W_CENTRALITY * centre


# ── Neural network evaluator ──────────────────────────────────────────────────

class NetEvaluator:
    """Wraps the Flax network for leaf evaluation and move ordering."""

    def __init__(self, params):
        self.params  = params
        self.has_net = params is not None

    def evaluate(self, board: Board) -> float:
        """
        Return a board value from the current player's perspective.
        Neural: tanh output × 50 to put it in the same scale as points.
        Fallback: hand-crafted heuristic.
        """
        if not self.has_net:
            return _heuristic(board)
        feat = NET.board_to_features(board)
        val, _ = NET.forward(self.params, feat)
        return float(val) * 50.0  # Scale to point range

    def ordered_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        """Re-order moves using policy-head probabilities (best first)."""
        if not self.has_net or len(moves) <= 1:
            return moves
        feat   = NET.board_to_features(board)
        _, logits = NET.forward(self.params, feat)
        mask   = NET.legal_move_mask(board)
        probs  = NET.policy_probs(logits, mask)
        scored = []
        for m in moves:
            try:   idx = NET.move_to_idx(m)
            except: idx = 0
            scored.append((probs[idx], m))  # Score by policy probability
        scored.sort(key=lambda x: -x[0])  # Sort descending
        return [m for _, m in scored]


# ── Move generation ────────────────────────────────────────────────────────────

def _gen_moves(board: Board) -> List[Move]:
    # Get valid moves, exclude carpet-1 (negative points), sort by priority (carpet > prime > plain)
    raw = board.get_valid_moves(exclude_search=True)
    result = [m for m in raw
              if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
    def key(m):
        if m.move_type == MoveType.CARPET:
            return 200 + CARPET_POINTS_TABLE.get(m.roll_length, 0)  # Higher for better carpets
        if m.move_type == MoveType.PRIME: return 100
        return 0
    result.sort(key=key, reverse=True)
    return result


# ── Negamax with neural network guidance ──────────────────────────────────────

class _Timeout(Exception): pass  # Exception for search timeout

class NeuralNegamax:
    """
    Negamax with alpha-beta + iterative deepening.
    Leaf evaluation and move ordering both use the neural network when available.
    """

    def __init__(self, evaluator: NetEvaluator):
        self.ev    = evaluator
        self.nodes = 0  # Node counter

    def best_move(self, board: Board, budget: float) -> Move:
        # Iterative deepening search within time budget
        self._dead = time.perf_counter() + budget
        moves = _gen_moves(board)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))  # Default search

        moves = self.ev.ordered_moves(board, moves)  # Order moves by policy
        best  = moves[0]

        for depth in range(1, _MAX_DEPTH + 1):
            if self._dead - time.perf_counter() < _MIN_SECS:
                break
            self.nodes = 0
            try:
                _, mv = self._negamax(board, depth, -INF, INF)
                if mv is not None:
                    best = mv
            except _Timeout:
                break
        return best

    def _negamax(self, board: Board, depth: int,
                 alpha: float, beta: float) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= self._dead:
            raise _Timeout()
        self.nodes += 1

        if board.is_game_over() or depth == 0:
            return self.ev.evaluate(board), None  # Leaf evaluation

        moves = _gen_moves(board)
        if not moves:
            return self.ev.evaluate(board), None

        if self.ev.has_net:
            moves = self.ev.ordered_moves(board, moves)  # Order moves

        best_val, best_move = -INF, None
        for m in moves:
            child = board.forecast_move(m, check_ok=True)  # Simulate move
            if child is None: continue
            child.reverse_perspective()  # Switch to opponent view
            child_val, _ = self._negamax(child, depth - 1, -beta, -alpha)
            val = -child_val
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if alpha >= beta:
                break  # Alpha-beta cutoff
        return best_val, best_move


# ── Player Agent ──────────────────────────────────────────────────────────────

class PlayerAgent:
    """
    Neural-network-guided negamax agent.

    No HMM / RatBelief. Rat searches are triggered by a simple EV threshold:
      EV = p * RAT_BONUS - (1-p) * RAT_PENALTY
    where p = 1/64 (uniform prior, no belief updates).  This gives
    EV ≈ -1.84, so searches are only made when the agent has no legal
    board moves — effectively disabling search in favour of pure carpeting.

    Once trained, the policy head naturally learns when to include searches.
    For the self-play training phase (using agent.py with HMM), the baseline
    agent handles rat search; this agent learns the carpeting strategy.
    """

    def __init__(self, board_state, transition_matrix=None,
                 time_left: Callable = None):
        # Load network params
        params = None
        global _EVAL_PARAMS
        if _EVAL_PARAMS is not None:
            params = _EVAL_PARAMS  # Use injected params for evaluation
        elif os.path.exists(NET_WEIGHTS_PATH):
            try:
                params = NET.load_params(NET_WEIGHTS_PATH)  # Load from disk
            except Exception as e:
                print(f"[agent_nn] weight load failed: {e} — using heuristic")

        self.evaluator = NetEvaluator(params)
        self.engine    = NeuralNegamax(self.evaluator)
        self.turn      = 0
        mode = "neural net" if self.evaluator.has_net else "heuristic fallback"
        print(f"[agent_nn] initialised ({mode})")

    def commentate(self) -> str:
        mode = "NN" if self.evaluator.has_net else "heuristic"
        return f"Turn {self.turn} | {mode} | nodes last turn: {self.engine.nodes}"

    def play(self, board_state: Board, sensor_data: Tuple,
             time_left: Callable) -> Move:
        self.turn += 1
        return self._decide(board_state, time_left)

    def _decide(self, board: Board, time_left: Callable) -> Move:
        turns_left  = board.player_worker.turns_left
        t_remaining = time_left() - _TIME_BUFFER

        if t_remaining <= 0:
            moves = _gen_moves(board)
            return moves[0] if moves else Move.search((0, 0))  # Timeout fallback

        # No HMM: skip rat searches (EV with uniform prior is always negative)
        # The agent focuses entirely on carpeting.

        per_turn = t_remaining / turns_left if turns_left > 0 else t_remaining
        budget   = max(0.05, min(5.0, per_turn * 0.8))  # Allocate time budget
        return self.engine.best_move(board, budget)
