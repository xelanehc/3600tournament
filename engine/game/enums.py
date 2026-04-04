from enum import IntEnum
from typing import Tuple

MAX_TURNS_PER_PLAYER = 40
BOARD_SIZE = 8
CARPET_POINTS_TABLE = {
    1: -1,
    2: 2,
    3: 4,
    4: 6,
    5: 10,
    6: 15,
    7: 21,
}
ALLOWED_TIME = 240
RAT_BONUS = 4
RAT_PENALTY = 2

class MoveType(IntEnum):
    PLAIN = 0
    PRIME = 1
    CARPET = 2
    SEARCH = 3

class Cell(IntEnum):
    SPACE = 0
    PRIMED = 1
    CARPET = 2
    BLOCKED = 3

class Noise(IntEnum):
    SQUEAK = 0
    SCRATCH = 1
    SQUEAL = 2

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

def loc_after_direction(loc: Tuple[int, int], dir: Direction) -> Tuple[int, int]:
    x, y = loc
    if dir == Direction.UP:
        return (x, y - 1)
    elif dir == Direction.DOWN:
        return (x, y + 1)
    elif dir == Direction.LEFT:
        return (x - 1, y)
    elif dir == Direction.RIGHT:
        return (x + 1, y)
    else:
        raise ValueError(f"Invalid direction:{dir}")

class Result(IntEnum):
    PLAYER = 0
    ENEMY = 1
    TIE = 2
    ERROR = 3

class ResultArbiter(IntEnum):
    PLAYER_A = 0
    PLAYER_B = 1
    TIE = 2
    ERROR = 3

class WinReason(IntEnum):
    POINTS = 0
    TIMEOUT = 1
    INVALID_TURN = 2
    CODE_CRASH = 3
    MEMORY_ERROR = 4
    FAILED_INIT = 5