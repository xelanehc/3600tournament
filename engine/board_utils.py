import random

from game.board import Board
from game.enums import Cell, MoveType, WinReason, BOARD_SIZE
from game.rat import Rat
import numpy as np

def get_board_string(board: Board, rat: Rat):
    """
    Returns a colored, bordered string representation of the current board state.
    Cells are 3 chars wide; borders are dimmed so content stands out.
    """
    RESET      = "\033[0m"
    DIM        = "\033[2;37m"      # dim gray  — borders and labels
    BG_PRIMED  = "\033[43m"        # yellow background
    BG_CARPET  = "\033[42m"        # green background
    BG_BLOCKED = "\033[100m"       # dark gray background
    FG_A       = "\033[1;94m"      # bold bright blue   (player A)
    FG_B       = "\033[1;91m"      # bold bright red    (player B)
    FG_RAT     = "\033[1;93m"      # bold bright yellow (rat)

    if board.player_worker.is_player_a:
        worker_a = board.player_worker
        worker_b = board.opponent_worker
    else:
        worker_a = board.opponent_worker
        worker_b = board.player_worker
    a_loc = worker_a.get_location()
    b_loc = worker_b.get_location()

    # Dimmed box-drawing pieces (3-wide cells)
    SEP = f"{DIM}│{RESET}"
    top = f"{DIM}  ┌{'───┬' * (BOARD_SIZE - 1)}───┐{RESET}"
    mid = f"{DIM}  ├{'───┼' * (BOARD_SIZE - 1)}───┤{RESET}"
    bot = f"{DIM}  └{'───┴' * (BOARD_SIZE - 1)}───┘{RESET}"

    lines = []

    # Column header (dimmed, aligned to 3-wide cells)
    lines.append(f"{DIM}   {''.join(f' {x}  ' for x in range(BOARD_SIZE))}{RESET}")
    lines.append(top)

    for y in range(BOARD_SIZE):
        row = [f"{DIM}{y} {RESET}{SEP}"]
        for x in range(BOARD_SIZE):
            loc = (x, y)
            cell_type = board.get_cell(loc)
            if cell_type == Cell.PRIMED:
                bg = BG_PRIMED
            elif cell_type == Cell.CARPET:
                bg = BG_CARPET
            elif cell_type == Cell.BLOCKED:
                bg = BG_BLOCKED
            else:
                bg = ""
            if loc == a_loc:
                row.append(f"{bg}{FG_A} A {RESET}{SEP}")
            elif loc == b_loc:
                row.append(f"{bg}{FG_B} B {RESET}{SEP}")
            elif rat.position == loc:
                row.append(f"{bg}{FG_RAT} r {RESET}{SEP}")
            elif cell_type == Cell.PRIMED:
                row.append(f"{BG_PRIMED} P {RESET}{SEP}")
            elif cell_type == Cell.CARPET:
                row.append(f"{BG_CARPET} C {RESET}{SEP}")
            elif cell_type == Cell.BLOCKED:
                row.append(f"{BG_BLOCKED}   {RESET}{SEP}")
            else:
                row.append(f"   {SEP}")
        lines.append("".join(row))
        if y < BOARD_SIZE - 1:
            lines.append(mid)

    lines.append(bot)
    return_string = "\n".join(lines) + "\n"

    return (
        return_string,
        worker_a.get_points(),
        worker_b.get_points(),
        worker_a.turns_left,
        worker_b.turns_left,
    )


def get_history_dict(board: Board, rat_position_history, spawn_a, spawn_b, errlog_a="", errlog_b=""):
    """
    Converts board history to a dictionary format for JSON serialization.

    Parameters:
        board: The game board with history
        rat_position_history: The final position of the rat
        errlog_a: Error log for player A
        errlog_b: Error log for player B

    Returns:
        Dictionary containing complete game history including:
        - pos: Position history for both players and rat at each turn
        - rat_caught: List of booleans indicating when rat was caught
        - Points, time, and turns tracking for both players
        - Move history and game metadata
    """
    board_hist = board.history
    history_dict = {
        "pos": board_hist.pos,
        "left_behind_enums": board_hist.left_behind_enums,
        "a_points": board_hist.a_points,
        "b_points": board_hist.b_points,
        "a_turns_left": board_hist.a_turns_left,
        "b_turns_left": board_hist.b_turns_left,
        "a_time_left": board_hist.a_time_left,
        "b_time_left": board_hist.b_time_left,
        "rat_caught": board_hist.rat_caught,
    }

    # Convert MoveType enums to strings
    left_behind = []
    for val in history_dict["left_behind_enums"]:
        match val:
            case MoveType.PLAIN:
                left_behind.append("plain")
            case MoveType.PRIME:
                left_behind.append("prime")
            case MoveType.CARPET:
                left_behind.append("carpet")
            case MoveType.SEARCH:
                left_behind.append("search")
            case _:
                left_behind.append("plain")
    history_dict["left_behind"] = left_behind
    history_dict.pop("left_behind_enums", None)

    # Add metadata
    history_dict["errlog_a"] = errlog_a
    history_dict["errlog_b"] = errlog_b
    history_dict["start_time"] = board.time_to_play
    history_dict["start_moves"] = board.MAX_TURNS
    history_dict["turn_count"] = board.turn_count
    history_dict["result"] = board.winner
    history_dict["reason"] = WinReason(board.win_reason).name

    history_dict["trapdoors"] = rat_position_history
    
    history_dict["spawn_a"] = spawn_a
    history_dict["spawn_b"] = spawn_b

    return history_dict


def get_history_json(board: Board, rat_position_history, spawn_a, spawn_b, err_a="", err_b=""):
    """
    Encodes the entire history of the game in a format readable by the renderer.

    Parameters:
        board: The game board with history
        rat_position_history: List of rat positions at each turn
        spawn_a: Spawn location for player A
        spawn_b: Spawn location for player B
        err_a: Error log for player A
        err_b: Error log for player B

    Returns:
        JSON string containing complete game history
    """
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            # JAX arrays: convert to numpy first, then fall through
            try:
                import jax.numpy as jnp
                if isinstance(obj, jnp.ndarray):
                    obj = np.asarray(obj)
            except ImportError:
                pass
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    return json.dumps(get_history_dict(board, rat_position_history, spawn_a, spawn_b, err_a, err_b), cls=NpEncoder)

def generate_spawns(board: Board):
    # Both players spawn in the center 4×4 (x=2–5, y=2–5), mirrored on the same row.
    x = random.randint(BOARD_SIZE // 2 - 2, BOARD_SIZE // 2 - 1)  # 2 or 3
    y = random.randint(BOARD_SIZE // 2 - 2, BOARD_SIZE // 2 + 1)  # center 4×4
    return (x, y), (BOARD_SIZE - 1 - x, y)
    