from .gridworld import ACTIONS, LabeledGridWorld, blind_craftsman, dungeon_quest
from .tasks import TASKS, BlindCraftsmanTask, DungeonQuestTask, Task

ENVS = {"dungeon_quest": dungeon_quest, "blind_craftsman": blind_craftsman}


def make(name: str):
    """Return the (environment, task) pair of a benchmark."""
    return ENVS[name](), TASKS[name]()


__all__ = ["ACTIONS", "LabeledGridWorld", "Task", "DungeonQuestTask", "BlindCraftsmanTask", "make"]
