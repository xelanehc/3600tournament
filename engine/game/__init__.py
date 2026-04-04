"""
This document provides a introduction to the documentation for how to play the game. Documentation for classes
available to players can be found to the left, and source code can be viewed from documentation as well.

**About Board**

Each turn, you will be given a `Board` (in `game.board`) instance, representing
a copy of the current game state without the rat.
You will also be given a callable `time_left()` function that, when called, will provide you
with the amount of time you have left for your turn in seconds. Rat data is held
by the game runner.

Both `apply_move` and `forecast_move` end a turn and pass to the next player.
Also note that `apply_move` and `forecast_move` do not automatically
reverse the perspective of the board - that is, functions will still call as if you are the player and your opponent is the
enemy. If you want to call methods for your opponent on the next turn call `Board.reverse_perspective()`.

Finally, remember that coordinates are returned in (x, y) form.

We use A and B rather than Black and White.  Player A goes first.

**Getting Started**

You will be creating a class called `PlayerAgent` that has a `play` method.  The state of the game will be passed as arguments to the `play` method.
Your `play` method will return a move object (see move.py)

Your `play` method will also get rat data.

You also have an optional 'commentate' method that you can use to print out debug statistics or other information at the end of the game. It should return a string. 

If you're lost about where to get started, we recommend that you take a look at
the Board functions, especially `apply_move`, `forecast_move`, and `get_valid_moves`,
alongside `apply_move` and `forecast_move` in the `Board` class.

**Extending the Board**

The `Board` class provides basic, high-level ways to interact with the board.
For the purposes of this assignment, Board methods may be sufficient for your needs
depending on how you design your bot.
You can write your own methods that use `Board` as a parameter for more complex functionality,
and you can even wrap the `Board` class in a class that you design.

You can also use methods from the underlying classes that `Board` wraps,
such as `Worker` and `Move`. It may be valuable to you to read through these some of these functions to understand
how the game runs at a lower level.

Modify internal state variables for Board
directly outside of functions at your own peril: you will have to understand
how the game operates through these functions, or else you risk modifying the board
in ways you did not anticipate. If you want to extend these classes or use the internal classes,
we recommend that you at least read the documentation and if you're curious read the
source code alongside it.
"""

import glob
import os

folder_path = os.path.dirname(__file__)  # or specify the path directly
py_files = glob.glob(os.path.join(folder_path, "*.py"))

# Extract the file names without the .py extension
module_names = [
    os.path.basename(f)[:-3] for f in py_files if os.path.basename(f) != "__init__.py"
]

# Set __all__ to the list of module names
__all__ = module_names
