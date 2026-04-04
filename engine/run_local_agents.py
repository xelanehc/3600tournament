import os
import pathlib
import sys
import time

from board_utils import get_history_json
from gameplay import play_game


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <player_a_name> <player_b_name>")
        sys.exit(1)

    sim_time = time.perf_counter()

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")

    player_a_name = sys.argv[1]
    player_b_name = sys.argv[2]

    final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b = play_game(
        play_directory,
        play_directory,
        player_a_name,
        player_b_name,
        display_game=True,
        delay=0.0,
        clear_screen=False,
        record=True,
        limit_resources=False,
    )


    sim_time = time.perf_counter() - sim_time
    turn_count = final_board.turn_count
    print(f"{sim_time:.2f} seconds elapsed, {turn_count} rounds.")

    records_dir = os.path.join(play_directory, "matches")
    os.makedirs(records_dir, exist_ok=True)
    i = 0
    while True:
        out_file = f"{player_a_name}_{player_b_name}_{i}.json"
        out_path = os.path.join(records_dir, out_file)
        if not os.path.exists(out_path):
            break
        i += 1

    with open(out_path, "w") as fp:
        fp.write(get_history_json(final_board, rat_position_history, spawn_a, spawn_b, message_a, message_b))


if __name__ == "__main__":
    main()