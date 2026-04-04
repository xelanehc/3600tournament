import random
from typing import List, Tuple
from .enums import *
from .worker import Worker
from .move import Move
from .history import History

class Board:
    """
    Board is the representation of the state of the match.

    Any coordinates should be given to the board in the form of x, y.

    Check_validity is on by default for most functions, but slows
    down execution. If a player is confident their actions are valid,
    they can directly apply turns and moves with check_validity as false.

    Be wary that invalid actions/turns could lead to functions throwing
    errors, so make sure to handle them with a try/except in case so that
    your program doesn't crash. If an apply function throws an error,
    it is not guarenteed that the board state will be valid or that the state
    will be the same as when the function started.
    """

    def __init__(
        self,
        time_to_play: float = 20,
        build_history: bool = False,
    ):
        """
        Initializes the board

        Parameters:
            time_to_play (float, optional): The time limit for the game in seconds. Defaults to 20.
            build_history (bool, optional): Whether to track the history of the game. Defaults to False.
        """
        # Board Metadata
        self.turn_count = 0
        self.is_player_a_turn = True
        self.winner = None
        self.time_to_play = time_to_play
        self.MAX_TURNS = MAX_TURNS_PER_PLAYER*2

        # Bitboard storage: Four 64-bit integers, one per cell type
        # Each bit corresponds to one cell: bit_index = y * BOARD_SIZE + x
        # Bit 0 = (0,0), Bit 1 = (1,0), Bit 8 = (0,1), Bit 63 = (7,7)
        # For ease of use, we have provided get_cell and set_cell functions that abstract away the bit manipulation.
        self._space_mask = 0xFFFFFFFFFFFFFFFF  # All cells start as SPACE
        self._primed_mask = 0x0000000000000000
        self._carpet_mask = 0x0000000000000000
        # Initialized by gameplay.py
        self._blocked_mask = 0x0000000000000000

        # Spawn position set by gameplay.py
        self.player_worker = Worker((-1, -1), is_player_a=True)
        self.opponent_worker = Worker((-1, -1), is_player_a=False)

        # Initialize worker time limits
        self.player_worker.time_left = time_to_play
        self.opponent_worker.time_left = time_to_play

        # History tracking
        self.build_history = build_history
        self.history = History() if build_history else None

        # Search information ((None, False)) if they did not guess)
        self.opponent_search = (None, False) # Last (Search Location, Search Result) for current opponent
        self.player_search = (None, False) # Last (Search Location, Search Result) for current player

        # precompute valid search moves
        self.valid_search_moves = [Move.search((x, y)) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)]

    def is_valid_move(
        self, move: Move, enemy: bool = False
    ):
        """
        Checks if a move is valid for the given player.

        Parameters:
            move (Move): The move to check.
            enemy (bool, optional): If True, checks the move for the enemy; if False, checks for the player. Defaults to False.

        Returns:
            (bool): True if the move is valid, False otherwise.
        """
        if enemy:
            worker = self.opponent_worker
        else:
            worker = self.player_worker

        my_loc = worker.get_location()

        match move.move_type:
            case MoveType.PLAIN:
                next_loc = loc_after_direction(my_loc, move.direction)
                return not self.is_cell_blocked(next_loc)

            case MoveType.PRIME:
                next_loc = loc_after_direction(my_loc, move.direction)
                if self.is_cell_blocked(next_loc):
                    return False
                # Can only prime if current cell is SPACE (not already PRIMED or CARPETED)
                bit_mask = 1 << self._loc_to_bit_index(my_loc)
                if (self._primed_mask | self._carpet_mask) & bit_mask:
                    return False
                return True

            case MoveType.CARPET:
                if move.roll_length < 1 or move.roll_length > BOARD_SIZE - 1:
                    return False

                # roll_length 1 = carpet the next square in the direction, etc.
                current_loc = my_loc
                for _ in range(1, move.roll_length + 1):
                    current_loc = loc_after_direction(current_loc, move.direction)

                    if not self.is_cell_carpetable(current_loc):
                        return False

                return True

            case MoveType.SEARCH:
                if not self.is_valid_cell(move.search_loc):
                    return False
                return True
            
        return False
        

    def get_valid_moves(self, enemy: bool = False, exclude_search = True) -> List[Tuple[Direction, MoveType]]:
        """
        Returns a list of all valid moves for the player or enemy.

        Parameters:
            enemy (bool, optional): If True, returns valid moves for enemy; if False, returns for player.
            exclude_search (bool, optional): If True, excludes search moves from the valid moves list. Defaults to True.

        Returns:
            [Move]: A list of valid moves for the specified player.
        """
        valid_moves = []
        worker = self.opponent_worker if enemy else self.player_worker
        my_loc = worker.get_location()
        my_bit = 1 << self._loc_to_bit_index(my_loc)

        # Pre-compute worker positions mask
        player_bit = 1 << self._loc_to_bit_index(self.player_worker.get_location())
        enemy_bit = 1 << self._loc_to_bit_index(self.opponent_worker.get_location())
        workers_mask = player_bit | enemy_bit

        # Pre-compute blocked cells mask (blocked OR primed OR workers)
        blocked_cells = self._blocked_mask | self._primed_mask | workers_mask

        # Pre-compute carpetable cells (primed cells not occupied by workers)
        carpetable_cells = self._primed_mask & ~workers_mask

        # Can only prime if current cell is SPACE (not already PRIMED or CARPETED)
        can_prime = not ((self._primed_mask | self._carpet_mask) & my_bit)

        # Check each direction with shift functions
        direction_shifts = [
            (Direction.UP, self._shift_mask_up),
            (Direction.DOWN, self._shift_mask_down),
            (Direction.LEFT, self._shift_mask_left),
            (Direction.RIGHT, self._shift_mask_right)
        ]

        for direction, shift_func in direction_shifts:
            # Check if next cell is blocked using bit shifting
            next_cell_mask = shift_func(my_bit)

            # PLAIN and PRIME moves - single bitwise AND instead of is_cell_blocked()
            if next_cell_mask and not (blocked_cells & next_cell_mask):
                valid_moves.append(Move.plain(direction))
                if can_prime:
                    valid_moves.append(Move.prime(direction))

            # CARPET moves - walk along ray checking primed cells
            current_mask = my_bit
            for roll_length in range(1, BOARD_SIZE):
                current_mask = shift_func(current_mask)

                # If shift resulted in empty mask, we've gone off the board
                if not current_mask:
                    break

                # Check if current cell is carpetable using single bitwise AND
                if not (carpetable_cells & current_mask):
                    break  # Can't carpet further in this direction

                valid_moves.append(Move.carpet(direction, roll_length))

        # Search Moves
        if not exclude_search:
            valid_moves.extend(self.valid_search_moves)

        return valid_moves
    
    def forecast_move(
        self,
        move: Move,
        check_ok: bool = True,
    ):
        """
        Creates a copy of the board with a forecasted move applied.

        Parameters:
            move (Move): The move to forecast.
            check_ok (bool, optional): If True, validates the move before applying; if False, skips validation. Defaults to True.

        Returns:
            (Board): A new Board object with the move applied, or None if the move is invalid.
        """
        board_copy = self.get_copy()
        ok = board_copy.apply_move(move, check_ok)
        return board_copy if ok else None

    def apply_move(
        self,
        move: Move,
        timer: float = 0,
        check_ok: bool = True,
    ):
        """
        Applies a move to the board.

        Parameters:
            move_type: The type of move (plain, prime, carpet, search).
            time (float, optional): Time taken for the move in seconds. Defaults to 0.
            check_ok (bool, optional): If True, validates the move before applying; if False, skips validation. Defaults to True.

        Returns:
            (bool): True if the move was successfully applied, False otherwise.
        """
        try:
            if check_ok:
                if not self.is_valid_move(move):
                    return False
                
            match move.move_type:
                case MoveType.PLAIN:
                    self.player_worker.position = loc_after_direction(self.player_worker.get_location(), move.direction)
                case MoveType.PRIME:
                    self.set_cell(self.player_worker.get_location(), Cell.PRIMED)
                    self.player_worker.position = loc_after_direction(self.player_worker.get_location(), move.direction)
                    self.player_worker.increment_points(amount=1)
                case MoveType.CARPET:
                    current_loc = self.player_worker.get_location()
                    for _ in range(1, move.roll_length + 1):
                        current_loc = loc_after_direction(current_loc, move.direction)
                        self.set_cell(current_loc, Cell.CARPET)

                    points = CARPET_POINTS_TABLE[move.roll_length]
                    self.player_worker.increment_points(amount=points)
                    self.player_worker.position = current_loc
                case MoveType.SEARCH:
                    # handled by game runner
                    pass
            
            self.end_turn(timer)
            
            return True
        except Exception as e:
            return False
            
    def end_turn(self, timer=0):
        """
        Ends the current turn and updates game state.
        Does NOT reverse perspective. 

        Parameters:
            timer (float, optional): Time taken for the turn in seconds. Defaults to 0.
        """
        self.turn_count += 1
        self.player_worker.turns_left -= 1
        self.player_worker.time_left -= timer

        self.check_win()

        self.is_player_a_turn = not self.is_player_a_turn
        
    def check_win(self, timeout_bounds: float = 0.5):
        """
        Checks if the game has been won and sets the winner accordingly.

        Parameters:
            timeout_bounds (float, optional): The time threshold in seconds for determining timeout ties. Defaults to 0.5.
        """
        if self.player_worker.time_left <= 0:
            if self.opponent_worker.time_left <= timeout_bounds:
                self.set_winner(Result.TIE, WinReason.TIMEOUT)
            else:
                self.set_winner(Result.ENEMY, WinReason.TIMEOUT)
        elif self.opponent_worker.time_left <= 0:
            if self.player_worker.time_left <= timeout_bounds:
                self.set_winner(Result.TIE, WinReason.TIMEOUT)
            else:
                self.set_winner(Result.PLAYER, WinReason.TIMEOUT)
        elif (self.player_worker.turns_left == 0 and self.opponent_worker.turns_left == 0) or self.turn_count >= 2 * MAX_TURNS_PER_PLAYER:
            if self.opponent_worker.get_points() > self.player_worker.get_points():
                self.set_winner(Result.ENEMY, WinReason.POINTS)
            elif self.opponent_worker.get_points() < self.player_worker.get_points():
                self.set_winner(Result.PLAYER, WinReason.POINTS)
            else:
                self.set_winner(Result.TIE, WinReason.POINTS)
                

    def is_game_over(self):
        """
        Checks if the game is over.

        Returns:
            (bool): True if the game is over, False otherwise.
        """
        return self.winner is not None


    def get_copy(
        self,
        build_history: bool = False,
    ):
        """
        Creates a deep copy of the current board.

        Parameters:
            build_history (bool, optional): Whether the copy should track history. Defaults to False.

        Returns:
            (Board): A new Board object with the same state as the current one.
        """

        new_board = Board(time_to_play=self.time_to_play, build_history=build_history)

        # Copy metadata
        new_board.turn_count = self.turn_count
        new_board.is_player_a_turn = self.is_player_a_turn
        new_board.winner = self.winner

        # Copy bitboards
        new_board._space_mask = self._space_mask
        new_board._primed_mask = self._primed_mask
        new_board._carpet_mask = self._carpet_mask
        new_board._blocked_mask = self._blocked_mask

        # Copy workers
        new_board.player_worker = self.player_worker.copy()
        new_board.opponent_worker = self.opponent_worker.copy()

        # Copy search info
        new_board.opponent_search = self.opponent_search
        new_board.player_search = self.player_search

        return new_board

    def set_winner(self, result: Result, reason: WinReason):
        """
        Sets the winner and the reason for the game's outcome.

        Parameters:
            result (Result): The winner of the game.
            reason (WinReason): The reason for the outcome.
        """

        self.winner = result
        self.win_reason = reason

    def get_winner(self) -> Result:
        """
        Returns the winner of the game.

        Returns:
            (Result): The winner of the game.
        """

        return self.winner

    def get_win_reason(self) -> str:
        """
        Returns the string explaining the reason why the game was won.

        Returns:
            (str): The reason for the game's outcome.
        """
        return self.win_reason

    def get_history(self) -> dict:
        """
        Get a dictionary representation for the renderer.

        Returns:
            (dict): A dictionary representing the game history.
        """
        return self.history

    def reverse_perspective(self):
        """
        Reverses the perspective from player to enemy or vice versa.
        This swaps all player and enemy references internally.
        """
        self.player_worker, self.opponent_worker = self.opponent_worker, self.player_worker

    def _loc_to_bit_index(self, loc: Tuple[int, int]) -> int:
        """
        Convert (x, y) coordinate to bit index in the 64-bit masks.

        Parameters:
            loc: Tuple of (x, y) coordinates

        Returns:
            int: Bit index (0-63) where bit_index = y * BOARD_SIZE + x
        """
        return loc[1] * BOARD_SIZE + loc[0]

    def _shift_mask_up(self, mask: int) -> int:
        """
        Shift mask up (decrease y). Clear bottom row.

        Returns:
            int: Shifted mask with bottom row cleared
        """
        return (mask >> BOARD_SIZE) & 0x00FFFFFFFFFFFFFF

    def _shift_mask_down(self, mask: int) -> int:
        """
        Shift mask down (increase y). Clear top row.

        Returns:
            int: Shifted mask with top row cleared
        """
        return (mask << BOARD_SIZE) & 0xFFFFFFFFFFFFFF00

    def _shift_mask_left(self, mask: int) -> int:
        """
        Shift mask left (decrease x). Clear rightmost column.

        Returns:
            int: Shifted mask with rightmost column cleared
        """
        return (mask >> 1) & 0x7F7F7F7F7F7F7F7F

    def _shift_mask_right(self, mask: int) -> int:
        """
        Shift mask right (increase x). Clear leftmost column.

        Returns:
            int: Shifted mask with leftmost column cleared
        """
        return (mask << 1) & 0xFEFEFEFEFEFEFEFE

        
    def get_cell(self, loc: Tuple[int, int]) -> Cell:
        """
        Returns the type of cell at the given coordinates.

        Parameters:
            (x, y): The location to check.

        Returns:
            (Cell): The type of cell at the specified location.
        """
        if not self.is_valid_cell(loc):
            raise ValueError(f"Invalid cell location: {loc}")

        bit_index = self._loc_to_bit_index(loc)
        bit_mask = 1 << bit_index

        # Check each mask to determine cell type
        if self._primed_mask & bit_mask:
            return Cell.PRIMED
        if self._carpet_mask & bit_mask:
            return Cell.CARPET
        if self._blocked_mask & bit_mask:
            return Cell.BLOCKED
        return Cell.SPACE
    
    def set_cell(self, loc: Tuple[int, int], cell_type: Cell):
        """
        Sets the type of cell at the given coordinates.

        Parameters:
            (x, y): The location to set.
            cell_type (Cell): The type of cell to set at the specified location.
        """
        if not self.is_valid_cell(loc):
            raise ValueError(f"Invalid cell location: {loc}")

        bit_index = self._loc_to_bit_index(loc)
        bit_mask = 1 << bit_index
        inv_mask = ~bit_mask

        # Clear the bit from all masks (ensures only one type set per cell)
        self._space_mask &= inv_mask
        self._primed_mask &= inv_mask
        self._carpet_mask &= inv_mask
        self._blocked_mask &= inv_mask

        # Set the bit in the appropriate mask
        if cell_type == Cell.SPACE:
            self._space_mask |= bit_mask
        elif cell_type == Cell.PRIMED:
            self._primed_mask |= bit_mask
        elif cell_type == Cell.CARPET:
            self._carpet_mask |= bit_mask
        elif cell_type == Cell.BLOCKED:
            self._blocked_mask |= bit_mask
        else:
            raise ValueError(f"Invalid cell type: {cell_type}")

    def is_valid_cell(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if the given coordinates are within the valid board boundaries.

        Parameters:
            (x, y): The location to check.

        Returns:
            (bool): True if the cell is valid, False otherwise.
        """
        return (
            loc[0] >= 0
            and loc[1] >= 0
            and loc[0] < BOARD_SIZE
            and loc[1] < BOARD_SIZE
        )
    
    def is_cell_blocked(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if the given cell is blocked to MOVEMENT.
        More specifically, checks if the cell is out of bounds, occupied by a worker, BLOCKED, or PRIMED.

        Parameters:
            (x, y): The location to check.
        Returns:
            (bool): True if the cell is blocked, False otherwise.
        """
        if not self.is_valid_cell(loc):
            return True

        enemy_loc = self.opponent_worker.get_location()
        if enemy_loc == loc:
            return True

        player_loc = self.player_worker.get_location()
        if player_loc == loc:
            return True

        bit_index = self._loc_to_bit_index(loc)
        bit_mask = 1 << bit_index

        return bool((self._blocked_mask | self._primed_mask) & bit_mask)
    
    def is_cell_carpetable(self, loc: Tuple[int, int]) -> bool:
        """
        Checks if the given cell can be carpeted.

        Parameters:
            (x, y): The location to check.
        Returns:
            (bool): True if the cell can be carpeted, False otherwise.
        """
        if not self.is_valid_cell(loc):
            return False

        enemy_loc = self.opponent_worker.get_location()
        if enemy_loc == loc:
            return False

        player_loc = self.player_worker.get_location()
        if player_loc == loc:
            return False

        bit_index = self._loc_to_bit_index(loc)
        bit_mask = 1 << bit_index

        return bool(self._primed_mask & bit_mask)



