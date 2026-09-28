"""Regular Positive and Negative Inference (RPNI).

Implements Algorithm 1 of the paper (Oncina & Garcia, 1992):

1. build a prefix tree acceptor (PTA) from the positive traces;
2. visit PTA states in length-lexicographic order and try to merge each one
   into an already-kept ("red") state; a merge is kept only if the resulting
   automaton still rejects every negative trace;
3. return the resulting DFA when no further merges are possible.

The implementation uses the standard red/blue formulation: merging a blue
state into a red state redirects the blue state's single incoming edge and
then folds the blue subtree into the red state to restore determinism.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .dfa import DFA, Symbol, reachable_renumbered

Trace = Tuple[Symbol, ...]
Delta = Dict[int, Dict[Symbol, int]]


def build_pta(positives: Iterable[Sequence[Symbol]]) -> Tuple[Delta, Set[int]]:
    """Prefix tree acceptor with states numbered in length-lexicographic order."""
    prefixes = {()}
    finals = set()
    for trace in positives:
        trace = tuple(trace)
        for i in range(1, len(trace) + 1):
            prefixes.add(trace[:i])
        finals.add(trace)

    ordered = sorted(prefixes, key=lambda p: (len(p), tuple(map(str, p))))
    index = {prefix: i for i, prefix in enumerate(ordered)}
    delta: Delta = {i: {} for i in range(len(ordered))}
    for prefix in ordered[1:]:
        delta[index[prefix[:-1]]][prefix[-1]] = index[prefix]
    return delta, {index[t] for t in finals}


def _accepts(delta: Delta, accepting: Set[int], trace: Sequence[Symbol]) -> bool:
    state = 0
    for symbol in trace:
        state = delta[state].get(symbol)
        if state is None:
            return False
    return state in accepting


def _merge(delta: Delta, accepting: Set[int], red: int, blue: int,
           parent: Tuple[int, Symbol]) -> Tuple[Delta, Set[int]]:
    """Return a copy of the automaton with ``blue`` merged into ``red``."""
    delta = {q: dict(edges) for q, edges in delta.items()}
    accepting = set(accepting)

    source, symbol = parent
    delta[source][symbol] = red

    def fold(target: int, subtree: int) -> None:
        if subtree in accepting:
            accepting.add(target)
        for sym, child in list(delta[subtree].items()):
            if sym in delta[target]:
                fold(delta[target][sym], child)
            else:
                delta[target][sym] = child
        del delta[subtree]
        accepting.discard(subtree)

    fold(red, blue)
    return delta, accepting


def _blue_states(delta: Delta, red: List[int]) -> List[Tuple[int, Tuple[int, Symbol]]]:
    """Children of red states that are not red, with their incoming edge."""
    red_set = set(red)
    blue = {}
    for q in red:
        for symbol, child in delta[q].items():
            if child not in red_set and child not in blue:
                blue[child] = (q, symbol)
    return sorted(blue.items())


def rpni(positives: Iterable[Sequence[Symbol]],
         negatives: Iterable[Sequence[Symbol]]) -> DFA:
    """Learn a DFA consistent with the positive and negative traces.

    Every positive trace is accepted and every negative trace is rejected by
    the returned automaton (under the strict, partial reading of the DFA).
    """
    positives = [tuple(t) for t in positives]
    negatives = [tuple(t) for t in negatives]
    overlap = set(positives) & set(negatives)
    if overlap:
        raise ValueError(f"Traces labelled both positive and negative: {sorted(overlap)[:3]}")

    delta, accepting = build_pta(positives)
    red = [0]

    while True:
        blue = _blue_states(delta, red)
        if not blue:
            break
        q_blue, parent = blue[0]
        for q_red in red:
            cand_delta, cand_accepting = _merge(delta, accepting, q_red, q_blue, parent)
            if not any(_accepts(cand_delta, cand_accepting, n) for n in negatives):
                delta, accepting = cand_delta, cand_accepting
                break
        else:
            red.append(q_blue)

    return reachable_renumbered(delta, 0, accepting)
