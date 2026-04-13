"""
Alan1 Agent Package

This package contains the Alan1 carpet game agent implementation,
split into separate modules for better organization.
"""

from .rat_belief import RatBelief
from .negamax import Negamax
from .agent import PlayerAgent

__all__ = ['RatBelief', 'Negamax', 'PlayerAgent']