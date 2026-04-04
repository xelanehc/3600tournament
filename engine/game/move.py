from .enums import MoveType, Direction

class Move:
    # __slots__ prevents the creation of a __dict__ for every instance, 
    # reduces memory usage and increases attribute access speed.
    __slots__ = ('move_type', 'direction', 'roll_length', 'search_loc')

    def __init__(self, move_type, direction=None, roll=0, search_loc=None):
        self.move_type = move_type
        self.direction = direction
        self.roll_length = roll
        self.search_loc = search_loc

    @classmethod
    def plain(cls, direction: Direction):
        """
        Create a plain move in the given direction.

        Parameters:
            direction (Direction): The direction to move.
        Example: 
            Move.plain(Direction.UP) creates a plain move in the UP direction.
        Returns:
            Move: A Move object representing the plain move.
        """
        return cls(MoveType.PLAIN, direction=direction)
    
    @classmethod
    def prime(cls, direction: Direction):
        """
        Create a prime move in the given direction.
        Parameters:
            direction (Direction): The direction to move.
        Example:
            Move.prime(Direction.LEFT) creates a prime move in the LEFT direction.
        Returns:
            Move: A Move object representing the prime move.
        """
        return cls(MoveType.PRIME, direction=direction)

    @classmethod
    def carpet(cls, direction: Direction, roll: int):
        """
        Create a carpet move in the given direction with the specified roll length.
        Parameters:
            direction (Direction): The direction to move.
            roll (int): The length of the carpet (1-7).
        Example: 
            Move.carpet(Direction.RIGHT, 3) creates a carpet move in the RIGHT direction with a roll length of 3.
        Returns:
            Move: A Move object representing the carpet move.
        """
        return cls(MoveType.CARPET, direction=direction, roll=roll)

    @classmethod
    def search(cls, search_loc = None):
        """
        Create a search move targeting the specified location.
        Parameters:
            search_loc (Tuple[int, int], optional): The location to search. Defaults to None.
        Example:
            Move.search((2, 3)) creates a search move targeting the location (2, 3)
        Returns:
            Move: A Move object representing the search move.
        """
        return cls(MoveType.SEARCH, search_loc=search_loc)
    
    def __repr__(self):
        if self.move_type == MoveType.PLAIN:
            return f"PLAIN({self.direction.name})"
        elif self.move_type == MoveType.PRIME:
            return f"PRIME({self.direction.name})"
        elif self.move_type == MoveType.CARPET:
            return f"CARPET({self.direction.name}, roll={self.roll_length})"
        elif self.move_type == MoveType.SEARCH:
            return f"SEARCH(loc={self.search_loc})"
        else:
            return "UNKNOWN_MOVE"