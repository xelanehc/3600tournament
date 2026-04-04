from collections import deque
import os
import pickle
import random
import time
from collections.abc import Iterable
import jax
import jax.numpy as jnp

def _load_transition_matrix():
    base_dir = os.path.join(os.path.dirname(__file__), "transition_matrices")
    pkl_files = [f for f in os.listdir(base_dir) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No transition matrices found in {base_dir}")
    pkl_path = os.path.join(base_dir, random.choice(pkl_files))

    with open(pkl_path, "rb") as f:
        T = pickle.load(f)
    T = jnp.asarray(T, dtype=jnp.float32)

    # Add up to 10% multiplicative noise per entry, then re-validate the matrix.
    key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
    noise = jax.random.uniform(key, T.shape, minval=-0.1, maxval=0.1)
    T = jnp.maximum(T * (1 + noise), 0)

    row_sum = T.sum(axis=1, keepdims=True)
    row_sum = jnp.where(row_sum == 0, 1.0, row_sum)
    T = T / row_sum

    return T

from board_utils import get_board_string, generate_spawns
from game.board import Board
from game.enums import *
from game.rat import Rat
from player_process import PlayerProcess
from game.enums import Cell, BOARD_SIZE


def init_display(board, player_a_name, player_b_name):
    # print(player_a_name+ " vs. "+player_b_name)
    # print("\nA to play" if board.is_player_a_turn else "\nB to play")
    pass


# prints board to terminal, clearing on each round
def print_board(board: Board, rat: Rat, clear_screen, board_only=False):
    player_map, a_points, b_points, a_turns, b_turns = get_board_string(board, rat)

    import os

    if clear_screen:
        if os.name == "nt":
            os.system("cls || clear")
        else:
            os.system("clear || cls")

    board_list = []
    if not board_only:
        board_list.append("\n--- TURN " + str(board.turn_count) + ": ")
        if board.is_player_a_turn:
            board_list.append(f"A to play, Time left:{board.player_worker.time_left:.2f}\n")
        else:
            board_list.append(f"B to play, Time left:{board.player_worker.time_left:.2f}\n")

    board_list.append(player_map)
    board_list.append(f" POINTS A:{a_points: <2d} B:{b_points: <2d}\n")
    board_list.append(f" TURNS  A:{a_turns: <2d} B:{b_turns: <2d}\n")

    print("".join(board_list), end="")


# prints a player's move on a given turn
def print_moves(player_as_turn, move, timer):
    try:
        if player_as_turn:
            print("A plays:", end="")
        else:
            print("B plays:", end="")
        if move is None:
            print("None", end="")
        else:
            if move.move_type == MoveType.CARPET:
                print(f"({move.direction.name}, {move.move_type.name}, roll={move.roll_length})", end="")
            elif move.move_type == MoveType.SEARCH:
                print(f"(SEARCH, loc={move.search_loc})", end="")
            else:
                print(f"({move.direction.name}, {move.move_type.name})", end="")
    except:
        print("Invalid", end="")

    print(f" in {timer:.3f} seconds")


def validate_submission(
    directory_a, player_a_name, limit_resources=False, use_gpu=False
):
    import traceback
    from multiprocessing import Queue

    player_a_process = None
    queues = []
    out_queue = None

    try:
        play_time = 360
        extra_ret_time = 5
        init_timeout = 10

        main_q = Queue()
        player_a_q = Queue()
        out_queue = Queue()
        queues = [player_a_q, main_q]

        T = _load_transition_matrix()

        rat = Rat(T)
        board = Board(play_time, build_history=False)
        spawn_a, spawn_b = generate_spawns(board)
        board.player_worker.position = spawn_a
        board.opponent_worker.position = spawn_b
        rat.spawn()

        player_a_process = PlayerProcess(
            True,
            player_a_name,
            directory_a,
            player_a_q,
            main_q,
            limit_resources,
            use_gpu,
            out_queue,
            user_name="player_a_user",
            group_name="player_a",
        )
        player_a_process.start()

        ok = main_q.get(block=True, timeout=10)
        message = ""

        if not ok:
            terminate_validation(player_a_process, queues, out_queue)
            return False, "Failed to initialize agent"

        player_a_process.pause_process_and_children()
        player_a_process.restart_process_and_children()
        ok, message = player_a_process.run_timed_constructor(
            board, init_timeout, extra_ret_time, T
        )
        player_a_process.pause_process_and_children()

        if ok:
            rat.move()
            samples = rat.sample(board)
            player_a_process.restart_process_and_children()
            move, _, message = player_a_process.run_timed_play(
                board, samples, board.player_worker.time_left, extra_ret_time
            )
            player_a_process.pause_process_and_children()
            ok = move is not None

        terminate_validation(player_a_process, queues, out_queue)
        return ok, message
    except:
        print(traceback.format_exc())
        if player_a_process is not None and player_a_process.process:
            terminate_validation(player_a_process, queues, out_queue)
        return False, traceback.format_exc()


def delete_module(name):
    import sys

    if name in sys.modules:
        del sys.modules[name]


def terminate_validation(process_a, queues, out_queue):
    delete_module("player_a.agent")
    delete_module("player_a")

    process_a.terminate_process_and_children()

    for q in queues:
        try:
            while True:
                q.get_nowait()
        except:
            pass

    try:
        while True:
            out_queue.get_nowait()
    except:
        pass


# Listener function to continuously listen to the queue
def listen_for_output(output_queue, stop_event):
    while not stop_event.is_set():
        try:
            print(output_queue.get(timeout=1))  # Wait for 1 second for output
        except:
            continue  # No output yet, continue listening


def play_game(
    directory_a,
    directory_b,
    player_a_name,
    player_b_name,
    display_game=False,
    delay=0,
    clear_screen=True,
    record=True,
    limit_resources=False,
    use_gpu=False,
):
    # setup main environment, import player modules
    import os
    import sys
    import threading
    import traceback
    from multiprocessing import Process, Queue

    if not directory_a in sys.path:
        sys.path.append(directory_a)

    if not directory_b in sys.path:
        sys.path.append(directory_b)

    play_time = 240
    extra_ret_time = 5
    init_timeout = 10

    if not limit_resources:
        init_timeout = 20
        play_time = 360

    # setup main thread queue for getting results
    main_q_a = Queue()
    main_q_b = Queue()

    # setup two thread queues for passing commands to players
    player_a_q = Queue()
    player_b_q = Queue()

    # game init

    T = _load_transition_matrix()

    rat = Rat(T)
    board = Board(play_time, build_history=record)
    # Random corner blockers: each corner independently gets 2×3, 3×2, or 2×2
    shapes = [(2, 3), (3, 2), (2, 2)]
    for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:  # TL, TR, BL, BR
        w, h = random.choice(shapes)
        for dx in range(w):
            for dy in range(h):
                x = dx if ox == 0 else BOARD_SIZE - 1 - dx
                y = dy if oy == 0 else BOARD_SIZE - 1 - dy
                board.set_cell((x, y), Cell.BLOCKED)

    spawn_a, spawn_b = generate_spawns(board)
    board.player_worker.position = spawn_a
    board.opponent_worker.position = spawn_b

    # Spawn rat to randomize initial position
    rat.spawn()
    rat_position_history = [rat.get_position()]

    out_queue = Queue()
    stop_event = None
    if not limit_resources:
        stop_event = threading.Event()
        listener_thread = threading.Thread(
            target=listen_for_output, args=(out_queue, stop_event)
        )
        listener_thread.daemon = True
        listener_thread.start()

    queues = [player_a_q, player_b_q, main_q_a, main_q_b]

    # startup two player processes
    player_a_process = PlayerProcess(
        True,
        player_a_name,
        directory_a,
        player_a_q,
        main_q_a,
        limit_resources,
        use_gpu,
        out_queue,
        user_name="player_a_user",
        group_name="player_a",
    )

    player_b_process = PlayerProcess(
        False,
        player_b_name,
        directory_b,
        player_b_q,
        main_q_b,
        limit_resources,
        use_gpu,
        out_queue,
        user_name="player_b_user",
        group_name="player_b",
    )

    success_a = False
    success_b = False

    message_a = ""
    message_b = ""

    

    try:
        player_a_process.start()
        success_a = main_q_a.get(block=True, timeout=10)
        player_a_process.pause_process_and_children()
    except Exception as e:
        message_a = traceback.format_exc()
        print(f"Player a crashed during initialization: {message_a}")

    try:
        player_b_process.start()
        success_b = main_q_b.get(block=True, timeout=10)
        player_b_process.pause_process_and_children()
    except Exception as e:
        message_b = traceback.format_exc()
        print(f"Player b crashed during initialization: {message_b}")

    if success_a and success_b:
        player_a_process.restart_process_and_children()
        success_a, message_a = player_a_process.run_timed_constructor(
            board, init_timeout, extra_ret_time, T
        )
        player_a_process.pause_process_and_children()

        player_b_process.restart_process_and_children()
        success_b, message_b = player_b_process.run_timed_constructor(
            board, init_timeout, extra_ret_time, T
        )
        player_b_process.pause_process_and_children()

    if not success_a and not success_b:
        board.set_winner(ResultArbiter.TIE, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return board, rat_position_history, spawn_a, spawn_b, message_a, message_b
    elif not success_a:
        board.set_winner(ResultArbiter.PLAYER_B, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return board, rat_position_history, spawn_a, spawn_b, message_a, message_b
    elif not success_b:
        board.set_winner(ResultArbiter.PLAYER_A, WinReason.FAILED_INIT)
        terminate_game(
            player_a_process, player_b_process, queues, out_queue, stop_event
        )
        return board, rat_position_history, spawn_a, spawn_b, message_a, message_b

    # start actual gameplay
    #
    timer = 0
    winner = ResultArbiter.TIE
    searches = deque([(None, False), (None, False)], maxlen=2) #(Search Loc, Search Result)
    while (not board.is_game_over()):
        if display_game:
            init_display(board, "PLAYER A", "PLAYER B")

        if display_game:
            print_board(
                board,
                rat,
                clear_screen,
            )


        # Rat moves every turn before sampling so the sensor always reflects
        # the rat's current position.
        rat.move()
        samples = rat.sample(board)

        if board.is_player_a_turn:
            # run a's turn
            player_label = "A"
            player_a_process.restart_process_and_children()
            move, timer, message_a = player_a_process.run_timed_play(
                board, samples, board.player_worker.time_left, extra_ret_time
            )
            player_a_process.pause_process_and_children()

        else:
            # run b's turn
            player_label = "B"
            player_b_process.restart_process_and_children()
            move, timer, message_b = player_b_process.run_timed_play(
                board, samples, board.player_worker.time_left, extra_ret_time
            )
            player_b_process.pause_process_and_children()

        if board.get_winner() is None:
            if move is None:
                if timer == -1:
                    board.set_winner(Result.ENEMY, WinReason.CODE_CRASH)
                elif timer == -2:
                    board.set_winner(Result.ENEMY, WinReason.MEMORY_ERROR)
                else:
                    board.set_winner(Result.ENEMY, WinReason.TIMEOUT)
                board.is_player_a_turn = not board.is_player_a_turn
            else:
                valid = board.apply_move(move, timer=timer, check_ok=True)

                if not valid:
                    if board.is_player_a_turn:
                        message_a = f"{move}"
                    else:
                        message_b = f"{move}"
                    board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
                    board.is_player_a_turn = not board.is_player_a_turn
                elif board.player_worker.time_left <= 0:
                    board.set_winner(Result.ENEMY, WinReason.TIMEOUT)


        # Check if rat was caught and track search info
        rat_was_caught = False
        search_loc = None
        search_result = None
        if move is not None and move.move_type == MoveType.SEARCH:
            search_loc = move.search_loc
            if (move.search_loc) == rat.get_position():
                rat_was_caught = True
                search_result = True
                rat.spawn()
                board.player_worker.increment_points(RAT_BONUS)
            else:
                search_result = False
                board.player_worker.decrement_points(RAT_PENALTY)

        searches.append((search_loc, search_result))

        # Record history after move is applied
        rat_position_history.append(rat.get_position())
        if move is not None and board.build_history and board.history:
            board.history.record_turn(board, move, rat_was_caught)

        if not move is None and display_game:
            print_moves(not board.is_player_a_turn, move, timer)
            time.sleep(delay)

        if not board.is_game_over():
            board.reverse_perspective()
            # After perspective reversal, set opponent search info for the next player
            board.opponent_search = searches[-1]
            board.player_search = searches[-2]

    win_result = board.get_winner()

    # might seem reversed but this is correct due to the way apply_move works
    if board.is_player_a_turn:
        if win_result == Result.PLAYER:
            winner = ResultArbiter.PLAYER_B
        elif win_result == Result.ENEMY:
            winner = ResultArbiter.PLAYER_A
    else:
        if win_result == Result.PLAYER:
            winner = ResultArbiter.PLAYER_A
        elif win_result == Result.ENEMY:
            winner = ResultArbiter.PLAYER_B

    board.set_winner(winner, board.win_reason)

    if board.is_game_over():
        if display_game:
            print_board(
                board,
                rat,
                clear_screen,
                board_only=True
            )
            print(f"{winner.name} wins by {board.get_win_reason().name}")
    
    if(message_a==""):
        message_a = player_a_process.run_timed_commentary(3)


    if(message_b==""):
        message_b = player_b_process.run_timed_commentary(3)
    print(f"Player A message: {message_a}")
    print(f"Player B message: {message_b}")
    terminate_game(player_a_process, player_b_process, queues, out_queue, stop_event)

    # Return board, spawns, and error messages
    return board, rat_position_history, spawn_a, spawn_b, message_a, message_b


# closes down player processes
def terminate_game(process_a, process_b, queues, out_queue, stop_event):
    delete_module("player_a" + "." + "agent")
    delete_module("player_a")
    delete_module("player_b" + "." + "agent")
    delete_module("player_b")

    if not stop_event is None:
        stop_event.set()
        try:
            while True:
                print(out_queue.get_nowait())
        except:
            pass

    process_a.terminate_process_and_children()
    process_b.terminate_process_and_children()

    for q in queues:
        try:
            while True:
                q.get_nowait()
        except:
            pass
