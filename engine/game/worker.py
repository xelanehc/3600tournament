from .enums import *

class Worker:
    def __init__(self, position: Tuple[int, int], is_player_a: bool):
        self.position = position
        self.is_player_a = is_player_a
        self.is_player_b = not is_player_a
        self.points = 0
        self.turns_left = MAX_TURNS_PER_PLAYER
        self.time_left = ALLOWED_TIME

    def get_location(self) -> Tuple[int, int]:
        """
        Get the current location of the worker.
        Returns:
            Tuple[int, int]: The (x, y) coordinates of the worker's current position.
        """
        return self.position
    
    def get_points(self) -> int:
        """
        Get the current points of the worker.
        Returns:
            int: The current points of the worker.
        """
        return self.points
    
    def increment_points(self, amount: int = 1):
        """
        Increment the worker's points by the specified amount.
        Parameters:
            amount (int): The amount to increment the points by. Defaults to 1.
        """
        self.points += amount

    def decrement_points(self, amount: int = 1):
        """
        Decrement the worker's points by the specified amount.
        Parameters:
            amount (int): The amount to decrement the points by. Defaults to 1.
        """
        self.points -= amount

    def copy(self):
        """
        Create a copy of the worker with the same attributes.
        Returns:
            Worker: A copy of the worker.
        """
        new_worker = Worker(self.position, self.is_player_a)
        new_worker.points = self.points
        new_worker.turns_left = self.turns_left
        new_worker.time_left = self.time_left
        return new_worker