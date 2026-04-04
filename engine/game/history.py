from typing import Tuple
from game.move import Move

class History:
    """Tracks the game history for replay and analysis."""
    def __init__(self):
        # List of board positions at each turn: [player_a_pos, player_b_pos, player_a_pos, ...]
        self.pos = []
        self.rat_pos = []
        # List of move types that left something behind: [MoveType.PLAIN, MoveType.PRIME, MoveType.CARPET, ...]
        self.left_behind_enums = []
        # Points scored over time
        self.a_points = []
        self.b_points = []
        # Turns remaining over time
        self.a_turns_left = []
        self.b_turns_left = []
        # Time remaining over time
        self.a_time_left = []
        self.b_time_left = []
        # Rat caught events (True if rat was caught on this turn)
        self.rat_caught = []

    def record_turn(self, board, move: Move, rat_caught: bool = False):
        """
        Record the state after a turn.

        Parameters:
            board: The game board
            move: The move that was made
            rat_caught: True if the rat was caught on this turn

        Note: This should be called AFTER apply_move (which calls end_turn and flips is_player_a_turn)
        but BEFORE reverse_perspective (which swaps workers).
        At this point: is_player_a_turn has been flipped, but workers haven't been swapped yet.
        """
        # Since end_turn has flipped is_player_a_turn, we need to flip our logic
        # If is_player_a_turn is False now, player A just moved (and is in player_worker)
        # If is_player_a_turn is True now, player B just moved (and is in player_worker)
        player_a_just_moved = not board.is_player_a_turn

        self.pos.append(board.player_worker.get_location())
        self.rat_caught.append(rat_caught)
        if player_a_just_moved:
            self.left_behind_enums.append(move.move_type)
            self.a_points.append(board.player_worker.get_points())
            self.b_points.append(board.opponent_worker.get_points())
            self.a_turns_left.append(board.player_worker.turns_left)
            self.b_turns_left.append(board.opponent_worker.turns_left)
            self.a_time_left.append(board.player_worker.time_left)
            self.b_time_left.append(board.opponent_worker.time_left)
        else:
            self.left_behind_enums.append(move.move_type)
            self.a_points.append(board.opponent_worker.get_points())
            self.b_points.append(board.player_worker.get_points())
            self.a_turns_left.append(board.opponent_worker.turns_left)
            self.b_turns_left.append(board.player_worker.turns_left)
            self.a_time_left.append(board.opponent_worker.time_left)
            self.b_time_left.append(board.player_worker.time_left)

