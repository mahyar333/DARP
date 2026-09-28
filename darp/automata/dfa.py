"""Deterministic finite automaton A = (Q, Sigma, delta, q0, F)."""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, Optional, Set, Tuple

Symbol = Hashable


class DFA:
    """DFA with integer states ``0..n-1`` and a possibly partial ``delta``.

    Two readings of a missing transition are used, as in the paper:

    * :meth:`accepts` (used during inference) rejects a trace that hits a
      missing transition, i.e. the automaton is partial.
    * :meth:`step` (used during planning) keeps the current state, i.e. labels
      that are irrelevant to the current sub-task are ignored.
    """

    def __init__(self, num_states: int, transitions: Dict[Tuple[int, Symbol], int],
                 start: int = 0, accepting: Iterable[int] = ()):
        self.num_states = num_states
        self.transitions = dict(transitions)
        self.start = start
        self.accepting: Set[int] = set(accepting)

    @property
    def states(self) -> range:
        return range(self.num_states)

    @property
    def alphabet(self) -> Set[Symbol]:
        return {symbol for (_, symbol) in self.transitions}

    def step(self, state: int, symbol: Optional[Symbol]) -> int:
        """Transition used on the product MDP; unknown symbols self-loop."""
        if symbol is None:
            return state
        return self.transitions.get((state, symbol), state)

    def run(self, trace: Iterable[Symbol]) -> Optional[int]:
        """Strict run: returns ``None`` if a transition is undefined."""
        state = self.start
        for symbol in trace:
            state = self.transitions.get((state, symbol))
            if state is None:
                return None
        return state

    def accepts(self, trace: Iterable[Symbol]) -> bool:
        return self.run(trace) in self.accepting

    def is_accepting(self, state: int) -> bool:
        return state in self.accepting

    def to_dot(self) -> str:
        """Graphviz DOT source (render with ``dot -Tpng dfa.dot -o dfa.png``)."""
        lines = ["digraph DFA {", "  rankdir=LR;", '  init [shape=point, label=""];']
        for q in self.states:
            shape = "doublecircle" if q in self.accepting else "circle"
            lines.append(f'  q{q} [shape={shape}, label="q{q}"];')
        lines.append(f"  init -> q{self.start};")
        edges: Dict[Tuple[int, int], list] = {}
        for (q, symbol), target in sorted(self.transitions.items(), key=str):
            edges.setdefault((q, target), []).append(str(symbol))
        for (q, target), symbols in edges.items():
            lines.append(f'  q{q} -> q{target} [label="{", ".join(symbols)}"];')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        edges = ", ".join(f"q{q}-{a}->q{t}" for (q, a), t in sorted(self.transitions.items(), key=str))
        return f"DFA(states={self.num_states}, start=q{self.start}, accepting={sorted(self.accepting)}, {edges})"


def reachable_renumbered(transitions: Dict[int, Dict[Symbol, int]], start: int,
                         accepting: Set[int]) -> DFA:
    """Build a :class:`DFA` from an adjacency map, numbering states in BFS order."""
    order = {start: 0}
    queue = deque([start])
    while queue:
        q = queue.popleft()
        for symbol in sorted(transitions.get(q, {}), key=str):
            target = transitions[q][symbol]
            if target not in order:
                order[target] = len(order)
                queue.append(target)
    delta = {
        (order[q], symbol): order[target]
        for q in order
        for symbol, target in transitions.get(q, {}).items()
    }
    return DFA(len(order), delta, start=0, accepting={order[q] for q in order if q in accepting})
