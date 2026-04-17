from collections import deque
from typing import Dict, List, Optional, Tuple
import time

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board

ALL_DIRS = list(Direction)
_MAX_DEPTH = 40
_MIN_SECS = 0.005
_BRANCH_EST = 3.8
_ASPIRATION_WINDOW = 1.5
_LMR_FULL_MOVES = 5
_LMR_MIN_DEPTH = 3
_FUTILITY_MARGINS = {1: 1.0, 2: 2.0}
INF = float('inf')

W_SCORE = 1.0
W_POTENTIAL = 0.48
W_RAT = 0.40
W_CENTRALITY = 0.10
W_OPP_RUN_DENY = 0.18
W_SPACE_SEEK = 0.12

_CENTRE = (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1)

_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_key(m: Move):
    return (m.move_type, m.direction, m.roll_length,
            m.search_loc if m.move_type == MoveType.SEARCH else None)


def _carpet_potential(board: Board, pos: Tuple[int, int],
                      other_pos: Tuple[int, int]) -> float:
    total = 0.0
    for d in ALL_DIRS:
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

        if primed_run >= 1:
            base_pts = CARPET_POINTS_TABLE.get(primed_run, 0)
            opp_steps = _dist(other_pos, pos)
            if opp_steps <= 1:
                base_pts *= 0.15
            elif opp_steps <= 3:
                base_pts *= 0.45
            elif opp_steps <= 5:
                base_pts *= 0.65
            if primed_run >= 3:
                base_pts *= 1.3
            total += base_pts

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
            total += CARPET_POINTS_TABLE.get(min(space_run, 7), 0) * 0.20
    return total


def _max_opp_primed_run(board: Board, opp_pos: Tuple[int, int],
                        my_pos: Tuple[int, int]) -> int:
    best = 0
    for d in ALL_DIRS:
        run = 0
        cur = opp_pos
        for _ in range(BOARD_SIZE - 1):
            cur = loc_after_direction(cur, d)
            if not board.is_valid_cell(cur):
                break
            if cur == opp_pos or cur == my_pos:
                break
            if board.get_cell(cur) == Cell.PRIMED:
                run += 1
            else:
                break
        if run > best:
            best = run
    return best


def _compute_space_target(board: Board) -> Tuple[int, int]:
    sx, sy, cnt = 0, 0, 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board.get_cell((x, y)) == Cell.SPACE:
                sx += x
                sy += y
                cnt += 1
    if cnt == 0:
        return _CENTRE
    return (round(sx / cnt), round(sy / cnt))


def _count_primed_neighbors(board: Board, loc: Tuple[int, int]) -> int:
    count = 0
    for d in ALL_DIRS:
        adj = loc_after_direction(loc, d)
        if board.is_valid_cell(adj) and board.get_cell(adj) == Cell.PRIMED:
            count += 1
    return count


def evaluate(board: Board, search_ev: float = 0.0) -> float:
    pw = board.player_worker
    ow = board.opponent_worker
    my_pos = pw.get_location()
    opp_pos = ow.get_location()

    score_delta = float(pw.get_points() - ow.get_points())

    turns_left = pw.turns_left
    frac_done = 1.0 - turns_left / MAX_TURNS_PER_PLAYER if MAX_TURNS_PER_PLAYER > 0 else 1.0

    my_pot = _carpet_potential(board, my_pos, opp_pos)
    opp_pot = _carpet_potential(board, opp_pos, my_pos)
    pot_scale = max(0.3, min(1.0, turns_left / 10.0))
    pot_delta = (my_pot - opp_pot) * pot_scale

    centrality_delta = float(_dist(opp_pos, _CENTRE) - _dist(my_pos, _CENTRE))

    opp_run = _max_opp_primed_run(board, opp_pos, my_pos)
    deny_penalty = CARPET_POINTS_TABLE.get(opp_run, 0) if opp_run >= 2 else 0.0

    space_target = _compute_space_target(board)
    space_delta = float(_dist(opp_pos, space_target) - _dist(my_pos, space_target))
    early_space_bias = max(0.0, 1.0 - frac_done)

    rat_term = search_ev * (1.0 + 0.5 * frac_done) if search_ev > 0.0 else 0.0

    return (W_SCORE * score_delta
            + W_POTENTIAL * pot_delta
            + W_CENTRALITY * centrality_delta
            - W_OPP_RUN_DENY * deny_penalty
            + W_SPACE_SEEK * space_delta * early_space_bias
            + W_RAT * rat_term)


def _gen_moves(board: Board, search_loc=None, search_ev=0.0,
               killers=None, history=None, tt_move_key=None,
               turns_left=40) -> List[Move]:
    raw = board.get_valid_moves(exclude_search=True)
    result = list(raw)

    if search_loc is not None and search_ev > 0.1:
        result.append(Move.search(search_loc))

    killer_set = set()
    if killers:
        killer_set = {_move_key(k) for k in killers if k is not None}

    hist = history or {}
    late_boost = 30 if turns_left <= 8 else 0

    def key(m):
        mk = _move_key(m)
        if tt_move_key is not None and mk == tt_move_key:
            return 10000

        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
            return 700 + 2 * pts + late_boost + hist.get(mk, 0)

        if mk in killer_set:
            return 400 + hist.get(mk, 0)

        if m.move_type == MoveType.SEARCH:
            return 520 + search_ev

        if m.move_type == MoveType.PRIME:
            next_loc = loc_after_direction(
                board.player_worker.get_location(), m.direction)
            primed_bonus = _count_primed_neighbors(board, next_loc) * 6
            return 200 + primed_bonus + hist.get(mk, 0)

        return 0 + hist.get(mk, 0)

    result.sort(key=key, reverse=True)
    return result


class _Timeout(Exception):
    pass


def _board_hash(board: Board) -> int:
    pw = board.player_worker
    ow = board.opponent_worker
    return hash((board._primed_mask, board._carpet_mask, board._blocked_mask,
                 pw.get_location(), ow.get_location(),
                 pw.get_points(), ow.get_points(),
                 pw.turns_left, ow.turns_left))


class Negamax:
    def __init__(self):
        self.nodes = 0
        self._deadline = 0.0
        self.last_depth = 0
        self._killers: Dict[int, List[Optional[Move]]] = {}
        self._history: Dict[tuple, float] = {}
        self._tt: Dict[int, Tuple[int, float, int, Optional[tuple]]] = {}

    def best_move(self, board: Board, budget: float,
                  pos_history: Optional[deque] = None,
                  search_loc=None, search_ev=0.0):
        self._deadline = time.perf_counter() + budget
        self._killers.clear()
        self._history.clear()
        self._tt.clear()
        self.nodes = 0

        turns_left = board.player_worker.turns_left
        moves = _gen_moves(board, search_loc, search_ev, turns_left=turns_left)
        if not moves:
            return Move.search((BOARD_SIZE // 2, BOARD_SIZE // 2))

        best = moves[0]
        prev_score = 0.0

        for depth in range(1, _MAX_DEPTH + 1):
            remaining = self._deadline - time.perf_counter()
            if remaining < _MIN_SECS:
                break

            iter_start = time.perf_counter()
            self.nodes = 0

            alpha = prev_score - _ASPIRATION_WINDOW
            beta = prev_score + _ASPIRATION_WINDOW

            try:
                score, mv = self._negamax(
                    board, depth, alpha, beta,
                    pos_history or deque(), search_loc, search_ev, True)

                if score <= alpha or score >= beta:
                    score, mv = self._negamax(
                        board, depth, -INF, INF,
                        pos_history or deque(), search_loc, search_ev, True)

                if mv is not None:
                    best = mv
                prev_score = score
                self.last_depth = depth

            except _Timeout:
                break

            iter_elapsed = time.perf_counter() - iter_start
            if iter_elapsed * _BRANCH_EST > (self._deadline - time.perf_counter()):
                break

        return best

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float,
                 pos_history: deque, search_loc=None, search_ev=0.0,
                 is_pv: bool = True) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= self._deadline:
            raise _Timeout()

        self.nodes += 1

        if board.is_game_over() or depth == 0:
            return evaluate(board, search_ev), None

        bh = _board_hash(board)
        tt_entry = self._tt.get(bh)
        tt_move_key = None
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_mk = tt_entry
            tt_move_key = tt_mk
            if tt_depth >= depth:
                if tt_flag == _TT_EXACT:
                    return tt_score, None
                elif tt_flag == _TT_LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == _TT_UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score, None

        static_eval = None
        if depth in _FUTILITY_MARGINS and not is_pv:
            static_eval = evaluate(board, search_ev)

        turns_left = board.player_worker.turns_left
        killers = self._killers.get(depth, [None, None])
        moves = _gen_moves(board, search_loc, search_ev,
                           killers, self._history, tt_move_key, turns_left)
        if not moves:
            return evaluate(board, search_ev), None

        best_val = -INF
        best_move = None
        orig_alpha = alpha

        for i, m in enumerate(moves):
            if (depth in _FUTILITY_MARGINS and not is_pv and i > 0
                    and m.move_type == MoveType.PLAIN and static_eval is not None
                    and static_eval + _FUTILITY_MARGINS[depth] <= alpha):
                continue

            child = board.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()

            use_lmr = (not is_pv
                        and depth >= _LMR_MIN_DEPTH
                        and i >= _LMR_FULL_MOVES
                        and m.move_type == MoveType.PLAIN)

            if i == 0:
                val, _ = self._negamax(child, depth - 1, -beta, -alpha,
                                       pos_history, search_loc, search_ev, is_pv)
                val = -val
            else:
                reduced_depth = depth - 2 if use_lmr else depth - 1
                val, _ = self._negamax(child, reduced_depth,
                                       -alpha - 0.001, -alpha,
                                       pos_history, search_loc, search_ev, False)
                val = -val
                if val > alpha and (use_lmr or val < beta):
                    val, _ = self._negamax(child, depth - 1, -beta, -alpha,
                                           pos_history, search_loc, search_ev,
                                           is_pv and val < beta)
                    val = -val

            if val > best_val:
                best_val = val
                best_move = m

            alpha = max(alpha, val)
            if alpha >= beta:
                mk = _move_key(m)
                self._history[mk] = self._history.get(mk, 0) + depth * depth
                if depth not in self._killers:
                    self._killers[depth] = [None, None]
                k = self._killers[depth]
                if k[0] is None or _move_key(k[0]) != mk:
                    k[1] = k[0]
                    k[0] = m
                break

        tt_flag = _TT_EXACT
        if best_val <= orig_alpha:
            tt_flag = _TT_UPPER
        elif best_val >= beta:
            tt_flag = _TT_LOWER
        bm_key = _move_key(best_move) if best_move is not None else None
        self._tt[bh] = (depth, best_val, tt_flag, bm_key)

        return best_val, best_move
