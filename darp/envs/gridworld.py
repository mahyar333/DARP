"""Labelled grid worlds with event traces.

The agent observes only its grid position. Entering a labelled cell emits the
cell's label as an event, and the sequence of events is the trace ``w`` on
which the non-Markovian reward ``R(w)`` is defined. Standing still, or
re-entering the cell that emitted the previous event, emits nothing: a cell
must be left for another labelled cell before it produces a new event.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

Position = Tuple[int, int]

ACTIONS: Tuple[str, ...] = ("up", "down", "left", "right")
_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class LabeledGridWorld:
    def __init__(self, size: int, labels: Dict[Position, str], obstacles: Iterable[Position] = (),
                 start: Position = (0, 0), terminal_labels: Iterable[str] = (), max_steps: int = 100,
                 name: str = "gridworld"):
        self.size = size
        self.labels = dict(labels)
        self.obstacles = set(obstacles)
        self.start = start
        self.terminal_labels = set(terminal_labels)
        self.max_steps = max_steps
        self.name = name
        self.actions = ACTIONS
        self.states: List[Position] = [(i, j) for i in range(size) for j in range(size)
                                       if (i, j) not in self.obstacles]

    @property
    def alphabet(self) -> List[str]:
        return sorted(set(self.labels.values()))

    def label(self, position: Position) -> Optional[str]:
        return self.labels.get(position)

    def move(self, position: Position, action: str) -> Position:
        dx, dy = _DELTAS[action]
        nxt = (position[0] + dx, position[1] + dy)
        if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size and nxt not in self.obstacles:
            return nxt
        return position

    def step(self, position: Position, action: str,
             last_event_cell: Optional[Position] = None) -> Tuple[Position, Optional[str], bool]:
        """Return (next position, emitted event or None, episode terminated)."""
        nxt = self.move(position, action)
        event = self.label(nxt) if nxt not in (position, last_event_cell) else None
        return nxt, event, event in self.terminal_labels


def dungeon_quest() -> LabeledGridWorld:
    """7x7 grid: key, chest (needs the key, yields the sword), shield, dragon (terminal)."""
    return LabeledGridWorld(
        size=7,
        labels={(2, 1): "key", (5, 5): "chest", (4, 2): "shield", (6, 6): "dragon"},
        terminal_labels={"dragon"}, max_steps=100, name="dungeon_quest")


def blind_craftsman() -> LabeledGridWorld:
    """7x7 grid with three wood sources, a factory, home (terminal) and two obstacles."""
    return LabeledGridWorld(
        size=7,
        labels={(1, 1): "wood", (1, 5): "wood", (5, 1): "wood", (3, 3): "factory", (5, 5): "home"},
        obstacles={(2, 2), (4, 4)},
        terminal_labels={"home"}, max_steps=80, name="blind_craftsman")
