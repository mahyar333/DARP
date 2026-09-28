"""Ground-truth non-Markovian objectives R: traces -> {0, 1}.

The learner never inspects these rules. It only observes the binary outcome
``R(w)`` of each episode. The rules are also used to evaluate which valid
behavioural *patterns* an agent has discovered.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

Trace = Tuple[str, ...]


class Task:
    name: str
    patterns: Dict[str, Trace]   # canonical trace of every valid pattern

    def initial(self):
        raise NotImplementedError

    def update(self, hidden, event: str):
        """Advance the hidden task state. Returns (new hidden state, reward, finished)."""
        raise NotImplementedError

    def reward(self, trace: Sequence[str]) -> int:
        hidden = self.initial()
        for event in trace:
            hidden, r, finished = self.update(hidden, event)
            if finished:
                return r
        return 0

    def pattern(self, trace: Sequence[str]) -> Optional[str]:
        """Name of the valid pattern a successful trace follows (None if unsuccessful)."""
        raise NotImplementedError


class DungeonQuestTask(Task):
    """Defeat the dragon holding the sword and the shield.

    The chest yields the sword only if the key was collected first. Reaching
    the dragon ends the episode. Three orderings of the sub-goals are valid.
    """

    name = "dungeon_quest"
    patterns = {
        "key-chest-shield": ("key", "chest", "shield", "dragon"),
        "key-shield-chest": ("key", "shield", "chest", "dragon"),
        "shield-key-chest": ("shield", "key", "chest", "dragon"),
    }

    def initial(self):
        return (False, False, False)  # key, sword, shield

    def update(self, hidden, event):
        key, sword, shield = hidden
        if event == "key":
            key = True
        elif event == "chest" and key:
            sword = True
        elif event == "shield":
            shield = True
        elif event == "dragon":
            return hidden, int(sword and shield), True
        return (key, sword, shield), 0, False

    def pattern(self, trace):
        if not self.reward(trace):
            return None
        firsts = []
        key = False
        for e in trace:
            if e == "key" and "key" not in firsts:
                firsts.append("key"); key = True
            elif e == "chest" and key and "chest" not in firsts:
                firsts.append("chest")
            elif e == "shield" and "shield" not in firsts:
                firsts.append("shield")
        return "-".join(firsts)


class BlindCraftsmanTask(Task):
    """Go home carrying exactly three tools.

    At most two units of wood can be carried; the factory turns all carried
    wood into tools (1:1). Waste-free production follows one of three batch
    patterns: 1+1+1, 2+1 or 1+2.
    """

    name = "blind_craftsman"
    capacity = 2
    required_tools = 3
    patterns = {
        "1+1+1": ("wood", "factory", "wood", "factory", "wood", "factory", "home"),
        "2+1": ("wood", "wood", "factory", "wood", "factory", "home"),
        "1+2": ("wood", "factory", "wood", "wood", "factory", "home"),
    }

    def initial(self):
        return (0, 0)  # wood carried, tools made

    def update(self, hidden, event):
        wood, tools = hidden
        if event == "wood":
            wood = min(self.capacity, wood + 1)
        elif event == "factory":
            tools, wood = tools + wood, 0
        elif event == "home":
            return hidden, int(tools == self.required_tools), True
        return (wood, tools), 0, False

    def pattern(self, trace):
        if not self.reward(trace):
            return None
        batches, wood = [], 0
        for e in trace:
            if e == "wood":
                wood = min(self.capacity, wood + 1)
            elif e == "factory" and wood:
                batches.append(str(wood)); wood = 0
        name = "+".join(batches)
        return name if name in self.patterns else None


TASKS = {"dungeon_quest": DungeonQuestTask, "blind_craftsman": BlindCraftsmanTask}
