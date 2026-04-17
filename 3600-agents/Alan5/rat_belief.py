from typing import Dict, Optional, Tuple
import numpy as np

from game.enums import Cell, BOARD_SIZE, RAT_BONUS, RAT_PENALTY
from game.board import Board

N = BOARD_SIZE * BOARD_SIZE

_NOISE_TABLE = np.array([
    [0.70, 0.15, 0.15],   # SPACE   = 0
    [0.10, 0.80, 0.10],   # PRIMED  = 1
    [0.10, 0.10, 0.80],   # CARPET  = 2
    [0.50, 0.30, 0.20],   # BLOCKED = 3
], dtype=np.float32)

_DIST_ERR_P   = np.array([0.12, 0.70, 0.12, 0.06], dtype=np.float32)
_DIST_ERR_OFF = np.array([-1,    0,    1,    2],    dtype=np.int32)
_MAX_DIST = 2 * (BOARD_SIZE - 1)

_NEIGHBOR_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_OPP_MISS_NEIGHBOR_BOOST = 1.15


def _idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _loc(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


def _build_dist_likelihood_table() -> np.ndarray:
    xs = np.arange(N, dtype=np.int32) % BOARD_SIZE
    ys = np.arange(N, dtype=np.int32) // BOARD_SIZE
    dist_mat = (np.abs(xs[:, None] - xs[None, :]) +
                np.abs(ys[:, None] - ys[None, :])).astype(np.int32)
    table = np.zeros((N, _MAX_DIST + 1, N), dtype=np.float32)
    for pos_idx in range(N):
        actual = dist_mat[pos_idx]
        adjusted = np.maximum(0, actual[:, None] + _DIST_ERR_OFF[None, :])
        for obs in range(_MAX_DIST + 1):
            table[pos_idx, obs] = (adjusted == obs).astype(np.float32) @ _DIST_ERR_P
    return table

_DIST_LK: np.ndarray = _build_dist_likelihood_table()


def _compute_spawn_prior(T: np.ndarray, steps: int = 1000) -> np.ndarray:
    origin_idx = _idx((0, 0))
    prior = np.zeros(N, dtype=np.float64)
    prior[origin_idx] = 1.0
    mat = np.linalg.matrix_power(T, steps)
    prior = prior @ mat
    s = prior.sum()
    if s > 0:
        prior /= s
    return prior


class RatBelief:
    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self._spawn_prior = _compute_spawn_prior(self.T)
        self.b = self._spawn_prior.copy()
        self._noise_cache: Dict[tuple, np.ndarray] = {}

    def predict(self):
        self.b = self.b @ self.T
        s = self.b.sum()
        if s > 0:
            self.b /= s

    def update_noise(self, board: Board, noise_type):
        ni = int(noise_type)
        lk = self._get_noise_lk(board)
        self.b *= lk[ni]
        self._norm()

    def update_distance(self, pos: Tuple[int, int], observed: int):
        pos_idx = _idx(pos)
        obs_clamped = min(observed, _MAX_DIST)
        self.b *= _DIST_LK[pos_idx, obs_clamped]
        self._norm()

    def update_search(self, loc: Tuple[int, int], found: bool,
                      is_opponent: bool = False):
        if found:
            self.b = self._spawn_prior.copy()
        else:
            self.b[_idx(loc)] = 0.0
            if is_opponent:
                lx, ly = loc
                for dx, dy in _NEIGHBOR_OFFSETS:
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        self.b[_idx((nx, ny))] *= _OPP_MISS_NEIGHBOR_BOOST
            self._norm()

    def best_search_ev(self) -> Tuple[Tuple[int, int], float]:
        i = int(np.argmax(self.b))
        p = float(self.b[i])
        return _loc(i), p * RAT_BONUS - (1.0 - p) * RAT_PENALTY

    def rat_ev(self) -> float:
        _, ev = self.best_search_ev()
        return ev

    def top_n(self, n: int = 3):
        idx = np.argsort(self.b)[-n:][::-1]
        return [(_loc(int(i)), float(self.b[i])) for i in idx]

    def _norm(self):
        s = self.b.sum()
        if s > 1e-15:
            self.b /= s
        else:
            self.b = self._spawn_prior.copy()

    def _get_noise_lk(self, board: Board) -> np.ndarray:
        key = (board._primed_mask, board._carpet_mask, board._blocked_mask)
        if key not in self._noise_cache:
            self._noise_cache[key] = self._build_noise_lk(board)
        return self._noise_cache[key]

    @staticmethod
    def _build_noise_lk(board: Board) -> np.ndarray:
        cell_arr = np.array([
            int(board.get_cell((i % BOARD_SIZE, i // BOARD_SIZE)))
            for i in range(N)
        ], dtype=np.int32)
        return _NOISE_TABLE[cell_arr].T.astype(np.float32)
