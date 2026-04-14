from collections import deque
from typing import List, Optional, Tuple
import time

from game.enums import Direction, MoveType, Cell, BOARD_SIZE, CARPET_POINTS_TABLE, loc_after_direction
from game.move import Move
from game.board import Board

ALL_DIRS = list(Direction)
_MAX_DEPTH = 100
_MIN_SECS = 0.005
INF = float('inf')

# Heuristic weights
W_SCORE = 1.0
W_POTENTIAL = 0.45
W_CENTRALITY = 0.10

_CENTRE = (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1)


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _carpet_potential(board: Board, pos: Tuple[int, int],
                      other_pos: Tuple[int, int]) -> float:
    """
    Opponent-aware carpet potential.
    For each direction, count primed runs and open-space lanes from pos.
    Discount primed runs if opponent is close enough to steal them.
    """
    total = 0.0

    for d in ALL_DIRS:
        # Pass 1: primed run
        primed_run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == pos or cur == other_pos:
                break
            if board.get_cell(cur) == Cell.PRIMED:
                primed_run += 1
            else:
                break

        if primed_run >= 2:
            base_pts = CARPET_POINTS_TABLE.get(primed_run, 0)
            # Discount if opponent can reach the roll position
            opp_steps = _dist(other_pos, pos)
            if opp_steps <= 1:
                base_pts *= 0.15
            elif opp_steps <= 3:
                base_pts *= 0.55
            total += base_pts

        # Pass 2: open space lane
        space_run = 0
        cur = pos
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == pos or cur == other_pos:
                break
            if board.get_cell(cur) == Cell.SPACE:
                space_run += 1
            else:
                break
        if space_run >= 2:
            total += CARPET_POINTS_TABLE.get(min(space_run, 7), 0) * 0.15

    return total


def _gen_moves(board: Board, search_loc=None, search_ev=0.0) -> List[Move]:
    raw = board.get_valid_moves(exclude_search=True)
    result = list(raw)

    if search_loc is not None and search_ev > 0.1:
        result.append(Move.search(search_loc))

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


def evaluate(board: Board) -> float:
    pw = board.player_worker
    ow = board.opponent_worker
    my_pos = pw.get_location()
    opp_pos = ow.get_location()

    score_delta = float(pw.get_points() - ow.get_points())

    my_pot = _carpet_potential(board, my_pos, opp_pos)
    opp_pot = _carpet_potential(board, opp_pos, my_pos)
    pot_delta = my_pot - opp_pot

    return W_SCORE * score_delta + W_POTENTIAL * pot_delta


class _Timeout(Exception):
    pass


class Negamax:
    def __init__(self):
        self.nodes = 0
        self._deadline = 0.0
        self.last_depth = 0

    def best_move(self, board: Board, budget: float, pos_history: Optional[deque] = None,
                  search_loc=None, search_ev=0.0):
        self._deadline = time.perf_counter() + budget
        moves = _gen_moves(board, search_loc, search_ev)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))

        best = moves[0]
        for depth in range(1, _MAX_DEPTH + 1):
            if self._deadline - time.perf_counter() < _MIN_SECS:
                break
            self.nodes = 0
            try:
                _, mv = self._negamax(board, depth, -INF, INF, pos_history or deque(), search_loc, search_ev)
                if mv is not None:
                    best = mv
                self.last_depth = depth
                #print(f"[Alan2] reached depth {depth} nodes={self.nodes}")
            except _Timeout:
                break
        return best

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float,
                  pos_history: deque, search_loc=None, search_ev=0.0) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= self._deadline:
            raise _Timeout()

        self.nodes += 1
        if board.is_game_over() or depth == 0:
            return evaluate(board), None

        moves = _gen_moves(board, search_loc, search_ev)
        if not moves:
            return evaluate(board), None

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