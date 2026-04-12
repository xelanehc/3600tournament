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
from collections import deque
from typing import List, Optional, Tuple, Dict
import time
import random

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
_TIME_BUFFER  = 0.15   # always keep this many seconds in reserve
_MAX_DEPTH    = 12     # iterative deepening ceiling (increased for deeper exploration)
_MIN_SECS     = 0.01   # don't start a new depth if less than this left

# Transposition table flags
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2


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

def _run_info(board, start, direction, p_pos, o_pos):
    """
    Returns (primed_run, space_run) in the given direction.
    primed_run stops at first non-PRIMED.
    space_run counts SPACE cells after primed_run.
    """
    primed = 0
    space  = 0
    cur = start

    # Count PRIMED
    for _ in range(BOARD_SIZE - 1):
        cur = loc_after_direction(cur, direction)
        if not board.is_valid_cell(cur) or cur in (p_pos, o_pos):
            return primed, space
        c = board.get_cell(cur)
        if c == Cell.PRIMED:
            primed += 1
        else:
            break

    # Count SPACE
    for _ in range(BOARD_SIZE - 1):
        if not board.is_valid_cell(cur) or cur in (p_pos, o_pos):
            break
        c = board.get_cell(cur)
        if c == Cell.SPACE:
            space += 1
            cur = loc_after_direction(cur, direction)
        else:
            break

    return primed, space


def _mobility(board, pos):
    moves = 0
    for d in ALL_DIRS:
        nxt = loc_after_direction(pos, d)
        if board.is_valid_cell(nxt) and not board.is_cell_blocked(nxt):
            moves += 1
    return moves

def _danger(board: Board, ppos: Tuple, opos: Tuple) -> float:
    """Penalty for being close to opponent."""
    d = _dist(ppos, opos)
    if d == 0: return -5.0
    if d == 1: return -3.0
    if d == 2: return -1.0
    return 0.0

def _contested_carpet_potential(board: Board, player_pos: Tuple, opponent_pos: Tuple) -> float:
    """Carpet potential considering opponent competition."""
    total_val = 0.0
    for d in ALL_DIRS:
        cur = player_pos
        primed = 0

        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur) or board.is_cell_blocked(cur):
                break

            c = board.get_cell(cur)

            if c == Cell.PRIMED:
                primed += 1
                continue

            # Check if opponent can contest this space
            if c == Cell.SPACE:
                d_me = _dist(player_pos, cur)
                d_opp = _dist(opponent_pos, cur)

                # If opponent is closer, skip
                if d_opp <= d_me:
                    break

                # Weight by accessibility
                potential_len = primed + 1
                weight = 1.0 / (1.0 + d_me)
                total_val += CARPET_POINTS_TABLE.get(min(potential_len, 7), 0) * weight
                break
            break

        # Immediate immediate carpet
        if primed >= 2:
            total_val += CARPET_POINTS_TABLE.get(primed, 0)

    return total_val

PHASE_WEIGHTS = {
    "opening": {"score": 0.2, "potential": 0.3, "mobility": 0.6, "danger": 0.3, "rat_ev": 0.1, "centrality": 0.4},
    "midgame": {"score": 1.0, "potential": 0.8, "mobility": 0.4, "danger": 0.2, "rat_ev": 0.3, "centrality": 0.2},
    "endgame": {"score": 1.5, "potential": 1.2, "mobility": 0.2, "danger": 0.5, "rat_ev": 0.8, "centrality": 0.1},
}

def evaluate(board: Board, belief: Optional['RatBelief'], pos_history: Optional[deque] = None) -> float:
    pw = board.player_worker
    ow = board.opponent_worker

    ppos = pw.get_location()
    opos = ow.get_location()

    # Identify phase
    turn = board.turn_count
    phase = "opening" if turn < 10 else "midgame" if turn < 30 else "endgame"
    W = PHASE_WEIGHTS[phase]

    # --- Core features -----------------------------------------------------

    # 1. Score difference
    score_delta = float(pw.get_points() - ow.get_points())

    # 2. Contested carpet potential
    my_pot  = _contested_carpet_potential(board, ppos, opos)
    opp_pot = _contested_carpet_potential(board, opos, ppos)
    pot_delta = my_pot - (opp_pot * 1.2)  # Slightly penalize opponent potential

    # 3. Mobility
    mob_delta = _mobility(board, ppos) - _mobility(board, opos)

    # 4. Danger from opponent
    danger = _danger(board, ppos, opos)

    # 5. Rat EV contribution
    rat_ev = belief.rat_ev() if belief is not None else 0.0

    # 6. Centrality
    centre_bonus = float(_dist(opos, _CENTRE) - _dist(ppos, _CENTRE))

    # 7. Repetition penalty (avoid cycles)
    repetition_penalty = 0.0
    if pos_history is not None:
        for i, prev_pos in enumerate(reversed(pos_history)):
            if ppos == prev_pos:
                repetition_penalty -= 1.0 / (i + 1)

    # --- Weighted sum ------------------------------------------------------

    return (
        W.get("score", 0)       * score_delta +
        W.get("potential", 0)   * pot_delta +
        W.get("mobility", 0)    * mob_delta +
        W.get("danger", 0)      * danger +
        W.get("rat_ev", 0)      * rat_ev +
        W.get("centrality", 0)  * centre_bonus +
        repetition_penalty
    )


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


class Zobrist:
    """Fast hashing for transposition table."""
    def __init__(self):
        random.seed(42)
        self.seed = random.getrandbits(64)

    def key(self, board: Board) -> int:
        """Create a hash key for the board state."""
        # Simple but effective: combine board state indicators
        k = (
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.turn_count % 2,
        )
        return hash(k)

class Negamax:
    """
    Negamax with alpha-beta pruning, transposition table, and quiescence.

    Convention: every call returns a value from the perspective of
    the CURRENT player (player_worker on the board passed in).
    """

    def __init__(self, belief: Optional[RatBelief]):
        self.belief   = belief
        self.nodes    = 0
        self._dead    = 0.0
        self.zobrist  = Zobrist()
        self.tt: Dict[int, Dict] = {}  # transposition table

    # ── Public entry ────────────────────────────────────────────────────

    def best_move(self, board: Board, budget: float, pos_history: Optional[deque] = None) -> Move:
        self._dead = time.perf_counter() + budget
        self.tt.clear()  # Reset TT for new search
        moves = _gen_moves(board)
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
        alpha_orig = alpha
        key = self.zobrist.key(board)

        # Transposition table lookup
        if key in self.tt:
            entry = self.tt[key]
            if entry['depth'] >= depth:
                if entry['flag'] == EXACT:
                    return entry['val'], entry['move']
                elif entry['flag'] == LOWERBOUND:
                    alpha = max(alpha, entry['val'])
                elif entry['flag'] == UPPERBOUND:
                    beta = min(beta, entry['val'])
                if alpha >= beta:
                    return entry['val'], entry['move']

        if board.is_game_over():
            val = evaluate(board, self.belief, pos_history)
            return val, None

        if depth == 0:
            # Use quiescence for leaf evaluation
            return self._quiescence(board, alpha, beta, pos_history), None

        moves = _gen_moves(board)
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

        # Store in transposition table
        flag = EXACT
        if best_val <= alpha_orig:
            flag = UPPERBOUND
        elif best_val >= beta:
            flag = LOWERBOUND

        self.tt[key] = {'depth': depth, 'val': best_val, 'flag': flag, 'move': best_move}

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
        moves = _gen_moves(board)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Player Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:
    """
    Negamax + HMM agent with position tracking and transposition table.

    Each turn:
    1. Update HMM with sensor data and any search results.
    2. If rat belief gives search EV above a dynamic threshold → Search.
    3. Otherwise → run negamax with iterative deepening, TT, and quiescence.
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
        # Budget: allocate more per-turn time for deeper exploration with TT efficiency.
        if turns_left > 0:
            per_turn = t_remaining / turns_left
        else:
            per_turn = t_remaining
        budget = max(0.10, min(6.0, per_turn * 0.95))

        return self.engine.best_move(board, budget, self.pos_history)
