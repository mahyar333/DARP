"""Confidence scores over automaton states (Eq. 1 of the paper).

    c(q) = (|P_q| + lambda |N_q|) / (|P_q| + |N_q| + beta)

``P_q`` / ``N_q`` count the positive / negative training traces whose run
passes through ``q``. A negative trace may leave the (partial) automaton;
it is counted on the prefix of its run that exists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from .automata import DFA


def run_states(dfa: DFA, trace: Sequence[str]) -> List[int]:
    """States visited by the strict run of ``trace`` (stops at a missing transition)."""
    q = dfa.start
    states = [q]
    for symbol in trace:
        q = dfa.transitions.get((q, symbol))
        if q is None:
            break
        states.append(q)
    return states


def confidence_scores(dfa: DFA, positives: Iterable[Sequence[str]], negatives: Iterable[Sequence[str]],
                      lam: float, beta: float) -> Dict[int, float]:
    pos, neg = defaultdict(int), defaultdict(int)
    for trace in positives:
        for q in set(run_states(dfa, trace)):
            pos[q] += 1
    for trace in negatives:
        for q in set(run_states(dfa, trace)):
            neg[q] += 1
    return {q: (pos[q] + lam * neg[q]) / (pos[q] + neg[q] + beta) for q in dfa.states}
