import random

from typing import Tuple
from .enums import Cell, Noise, BOARD_SIZE

HEADSTART_MOVES = 1000

# Noise probabilities based on cell type
# [P(squeak), P(scratch), P(squeal)]
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

# Distance estimation error probabilities
# [P(-1), P(correct), P(+1), P(+2)]

# DEV NOTE: Distance error probs seems to have the largest effect on rat search accuracy
DISTANCE_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)
DISTANCE_ERROR_OFFSETS = (-1, 0, 1, 2)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def cumulative(probs):
    c = []
    s = 0.0
    for p in probs:
        s += p
        c.append(s)
    return c


class Rat:
    def __init__(self, T):
        """
        T: (BOARD_SIZE*BOARD_SIZE) x (BOARD_SIZE*BOARD_SIZE) transition matrix
        T[from_index, to_index] = probability of moving from position `from_index` to position `to_index`
        where index = y * BOARD_SIZE + x for position (x, y)
        Each row sums to 1.0 (valid probability distribution)
        """

        self.T = T

        # Precompute cumulative probabilities for each starting position
        # cumT[i] = cumulative distribution for row i
        num_positions = BOARD_SIZE * BOARD_SIZE

        self.cumT = [[0] * len(T[0]) for _ in range(num_positions)]

        for i in range(num_positions):
            running_sum = 0
            for j in range(len(T[i])):
                running_sum += T[i][j]
                self.cumT[i][j] = running_sum

        self.noise_cum = {k: cumulative(v) for k, v in NOISE_PROBS.items()}
        self.dist_cum = cumulative(DISTANCE_ERROR_PROBS)

        self.position = (0, 0)


    def _pos_to_index(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to index in transition matrix."""
        return pos[1] * BOARD_SIZE + pos[0]

    def _index_to_pos(self, index: int) -> Tuple[int, int]:
        """Convert transition matrix index to (x, y) position."""
        y = index // BOARD_SIZE
        x = index % BOARD_SIZE
        return (x, y)

    def _sample3(self, cum):
        r = random.random()
        if r < cum[0]: return 0
        if r < cum[1]: return 1
        return 2

    def move(self):
        """Move the rat according to the transition matrix."""
        # Convert current position to index
        from_index = self._pos_to_index(self.position)

        # Sample from cumulative distribution for this position
        r = random.random()
        cum_probs = self.cumT[from_index]

        # Binary search or linear search to find which position we transition to
        # Linear search is fine for 64 positions
        to_index = 0
        for i in range(len(cum_probs)):
            if r < cum_probs[i]:
                to_index = i
                break

        # Convert back to (x, y) position
        self.position = self._index_to_pos(to_index)

    def make_noise(self, board) -> Noise:
        """Generate noise based on the cell type at the rat's position."""
        cell_type = board.get_cell(self.position)

        cum = self.noise_cum.get(cell_type, self.noise_cum[Cell.SPACE])
        idx = self._sample3(cum)

        return Noise(idx)

    def estimate_distance(self, worker_position: Tuple[int, int]) -> int:
        actual = manhattan_distance(worker_position, self.position)

        r = random.random()
        cum = self.dist_cum

        offset = DISTANCE_ERROR_OFFSETS[-1]
        for i, threshold in enumerate(cum):
            if r < threshold:
                offset = DISTANCE_ERROR_OFFSETS[i]
                break

        d = actual + offset
        return d if d > 0 else 0

    def spawn(self):
        self.position = (0,0)

        for _ in range(HEADSTART_MOVES):
            self.move()

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def sample(self, board):
        """Generate noise and distance samples for the current turn."""
        return (
            self.make_noise(board),
            self.estimate_distance(board.player_worker.get_location()),
        )