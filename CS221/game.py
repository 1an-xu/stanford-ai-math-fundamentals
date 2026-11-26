from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

# Define core type aliases
State = Any
Action = str
Policy = Callable[[State], Dict[Action, float]]

class Game:

    def start_state(self) -> State:
        raise NotImplementedError

    def successors(self, state: State) -> Dict[Action, State]:
        raise NotImplementedError

    def is_end(self, state: State) -> bool:
        raise NotImplementedError
    
    def player(self, state: State) -> str:
        raise NotImplementedError

    def utility(self, state: State) -> float:
        raise NotImplementedError

class Game1(Game):
    """Simple discrete game tree from Lecture 10."""
    def start_state(self) -> State:
        return "root"

    def successors(self, state: State) -> Dict[Action, State]:
        mapping = {
            "root": {"A": "A", "B": "B", "C": "C"},
            "A": {"1": "A1", "2": "A2"},
            "B": {"1": "B1", "2": "B2"},
            "C": {"1": "C1", "2": "C2"},
        }
        return mapping[state]

    def player(self, state: State) -> str:
        if state == "root":
            return "agent"
        if state in ["A", "B", "C"]:
            return "opp"
        raise ValueError(f"Invalid state:{state}")

    def is_end(self, state: State) -> bool:
        return state in ["A1", "A2", "B1", "B2", "C1", "C2"]

    def utility(self, state: State) -> float:
        utilities = {
            "A1": -50, "A2": 50,
            "B1": 1,   "B2": 3,
            "C1": -5,  "C2": 15,
        }
        return utilities[state]

@dataclass(frozen=True)
class HalvingState:
    n: int
    player: str

class HalvingGame(Game):
    """
    Halving Game: Two players take turns decrementing (-1) or halving (//2) a number.
    The player who reaches 0 wins.
    """
    def __init__(self, n: int):
        self.n = n

    def start_state(self) -> HalvingState:
        return HalvingState(n=self.n, player="agent")

    def successors(self, state: HalvingState) -> Dict[Action, HalvingState]:
        next_player = "opp" if state.player == "agent" else "agent"
        return {
            "decrement": HalvingState(n=state.n - 1, player=next_player),
            "half": HalvingState(n=state.n // 2, player=next_player),
        }

    def player(self, state: HalvingState) -> str:
        return state.player

    def is_end(self, state: HalvingState) -> bool:
        return state.n == 0

    def utility(self, state: HalvingState) -> float:
        assert state.n == 0
        # If it's agent's turn when n=0, it means opponent made the move that resulted in 0 -> Opponent wins!
        # If it's opp's turn, agent made the move that resulted in 0 -> Agent wins!
        return -1.0 if state.player == "agent" else 1.0