"""
Rat Belief Tracking using Hidden Markov Model.

This module implements the RatBelief class for tracking the probability
distribution of the rat's position based on sensor data and search results.
"""

from typing import Tuple
import numpy as np

from game.enums import Cell, BOARD_SIZE, RAT_BONUS, RAT_PENALTY
from game.board import Board


# Constants
N = BOARD_SIZE * BOARD_SIZE

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


# Helpers
def _idx(loc: Tuple[int,int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]

def _loc(idx: int) -> Tuple[int,int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)

def _dist(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


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