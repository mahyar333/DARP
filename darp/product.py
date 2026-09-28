"""Product MDP bookkeeping and Q-value transfer after a refinement (Sec. 2.3)."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from .automata import DFA


def access_words(dfa: DFA) -> Dict[int, tuple]:
    """Shortest (length-lexicographic) word reaching every state."""
    out = {dfa.start: ()}
    queue = deque([dfa.start])
    edges = sorted(dfa.transitions.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
    while queue:
        q = queue.popleft()
        for (src, symbol), dst in edges:
            if src == q and dst not in out:
                out[dst] = out[q] + (symbol,)
                queue.append(dst)
    return out


def state_correspondence(old: DFA, new: DFA) -> Dict[int, Optional[int]]:
    """Map every new state to the old state reached by the same access word (or None)."""
    return {q: old.run(word) for q, word in access_words(new).items()}


def distance_to_accept(dfa: DFA) -> Dict[int, int]:
    """Number of transitions from each state to the nearest accepting state."""
    reverse: Dict[int, set] = {}
    for (src, _), dst in dfa.transitions.items():
        reverse.setdefault(dst, set()).add(src)
    dist = {q: 0 for q in dfa.accepting}
    queue = deque(dfa.accepting)
    while queue:
        q = queue.popleft()
        for p in reverse.get(q, ()):
            if p not in dist:
                dist[p] = dist[q] + 1
                queue.append(p)
    return dist


def transfer_q_values(q_old: Dict, old: DFA, new: DFA, positions, num_actions: int,
                      eps_init: float) -> Dict:
    """Q_{t+1} from Q_t following the paper's three cases.

    * product states whose automaton state is unchanged keep their values;
    * product states of newly created automaton states are initialised with
      the average over the old product states with the same position;
    * product states of changed automaton states (Delta Q) get ``eps_init``.
    """
    mapping = state_correspondence(old, new)
    old_edges = {q: {} for q in old.states}
    for (src, symbol), dst in old.transitions.items():
        old_edges[src][symbol] = dst
    new_edges = {q: {} for q in new.states}
    for (src, symbol), dst in new.transitions.items():
        new_edges[src][symbol] = dst

    q_new = {}
    for qn in new.states:
        qo = mapping.get(qn)
        if qo is None:  # new automaton state
            for s in positions:
                values = [q_old[(s, x)] for x in old.states if (s, x) in q_old]
                if values:
                    q_new[(s, qn)] = np.mean(values, axis=0)
            continue
        unchanged = (
            new.is_accepting(qn) == old.is_accepting(qo)
            and set(new_edges[qn]) == set(old_edges[qo])
            and all(mapping.get(new_edges[qn][a]) == old_edges[qo][a] for a in new_edges[qn])
        )
        for s in positions:
            if unchanged and (s, qo) in q_old:
                q_new[(s, qn)] = q_old[(s, qo)].copy()
            elif not unchanged:
                q_new[(s, qn)] = np.full(num_actions, eps_init)
    return q_new
