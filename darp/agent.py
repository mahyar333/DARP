"""DARP: Dynamic Automaton Refinement and Planning.

One agent class implements DARP and, by switching components off, the
passive baselines used in the paper:

=============  ===========  ==================  =================
method         refinement   exploration bonus   initial positives
=============  ===========  ==================  =================
DARP           yes          yes                 as configured
Static RPNI    no           no                  none
HiPO           no           no                  one known pattern
=============  ===========  ==================  =================
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .automata import DFA, rpni
from .confidence import confidence_scores, run_states
from .envs import LabeledGridWorld, Task
from .product import distance_to_accept, state_correspondence, transfer_q_values

Trace = Tuple[str, ...]


@dataclass
class DARPConfig:
    # Online refinement
    refine: bool = True
    coverage_gap: bool = True        # also refine when the trace visits a state with c(q) < theta_min
    theta_min: float = 0.2
    lam: float = 1.0                 # weight of negative evidence in c(q)
    beta: float = 2.0                # smoothing in c(q)
    # Exploration bonus (Eq. 2): alpha_bonus * sqrt(log t / N(q)) * max(1 - c(q), 0)
    exploration_bonus: bool = True
    alpha_bonus: float = 0.5
    # Q-value transfer after a refinement
    q_transfer: bool = True
    eps_init: float = 0.1
    # Q-learning on the product MDP
    alpha: float = 0.5
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.992
    goal_reward: float = 1.0
    progress_reward: float = 0.0     # scale of potential-based shaping with Phi(q) = -distance to acceptance
    step_penalty: float = 0.02


@dataclass
class EpisodeLog:
    success: List[int] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    refined: List[int] = field(default_factory=list)
    dfa_states: List[int] = field(default_factory=list)
    patterns_known: List[int] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    queries: List[int] = field(default_factory=list)   # cumulative oracle queries (AGI only)
    samples: List[int] = field(default_factory=list)   # cumulative environment steps, incl. those spent answering queries


def with_prefixes(traces: Iterable[Trace], positives: Set[Trace]) -> Set[Trace]:
    """Every strict prefix of an observed trace received reward 0."""
    out = set()
    for w in traces:
        out.update(w[:i] for i in range(len(w)) if w[:i] not in positives)
    return out


class DARPAgent:
    def __init__(self, env: LabeledGridWorld, task: Task, cfg: DARPConfig,
                 positives: Iterable[Sequence[str]] = (), negatives: Iterable[Sequence[str]] = (),
                 seed: int = 0):
        self.env, self.task, self.cfg = env, task, cfg
        self.rng = np.random.default_rng(seed)
        self.num_actions = len(env.actions)
        self.P: Set[Trace] = {tuple(w) for w in positives}
        self.N: Set[Trace] = {tuple(w) for w in negatives}
        self.N |= with_prefixes(self.P | self.N, self.P)
        self.initial_patterns = {task.pattern(w) for w in self.P} - {None}

        self.q: Dict = defaultdict(lambda: np.zeros(self.num_actions))
        self.visits: Dict[int, int] = defaultdict(int)
        self.epsilon = cfg.epsilon_start
        self.episode_index = 0
        self.refinements = 0
        self.log = EpisodeLog()
        self._start = time.perf_counter()
        self._synthesize()

    # ------------------------------------------------------------------ automaton
    def _synthesize(self) -> None:
        self.dfa: DFA = rpni(sorted(self.P), sorted(self.N))
        self.confidence = confidence_scores(self.dfa, self.P, self.N, self.cfg.lam, self.cfg.beta)
        self.distance = distance_to_accept(self.dfa)

    def _refine(self, trace: Trace, outcome: int) -> None:
        old_dfa, old_visits = self.dfa, self.visits
        (self.P if outcome else self.N).add(trace)
        self.N |= with_prefixes([trace], self.P)
        self._synthesize()
        self.refinements += 1

        mapping = state_correspondence(old_dfa, self.dfa)
        self.visits = defaultdict(int, {q: old_visits.get(mapping[q], 0)
                                        for q in self.dfa.states if mapping.get(q) is not None})
        if self.cfg.q_transfer:
            self.q = defaultdict(lambda: np.zeros(self.num_actions),
                                 transfer_q_values(self.q, old_dfa, self.dfa, self.env.states,
                                                   self.num_actions, self.cfg.eps_init))
        else:
            self.q = defaultdict(lambda: np.zeros(self.num_actions))

    def patterns_known(self) -> Set[str]:
        """Valid patterns accepted by the current automaton beyond the initial knowledge."""
        return {name for name, w in self.task.patterns.items()
                if self.dfa.accepts(w)} - self.initial_patterns

    # ------------------------------------------------------------------ acting
    def _bonus(self, q: int) -> float:
        if not self.cfg.exploration_bonus:
            return 0.0
        t = max(self.episode_index, 2)
        return (self.cfg.alpha_bonus * math.sqrt(math.log(t) / max(1, self.visits[q]))
                * max(1.0 - self.confidence.get(q, 0.0), 0.0))

    def _potential(self, q: int) -> float:
        # Unreachable states get the largest distance + 1 so shaping stays bounded.
        far = max(self.distance.values(), default=0) + 1
        return -float(self.distance.get(q, far))

    def _reward(self, q: int, q_next: int) -> float:
        """Goal reward, step cost and potential-based shaping (Ng et al., 1999).

        The undiscounted form Phi(q') - Phi(q) telescopes over an episode, so
        cycling through the automaton cannot accumulate shaping reward.
        """
        r = self.cfg.goal_reward if self.dfa.is_accepting(q_next) else -self.cfg.step_penalty
        return r + self.cfg.progress_reward * (self._potential(q_next) - self._potential(q))

    def _act(self, state, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def run_episode(self, learn: bool = True) -> Tuple[int, Trace]:
        self.episode_index += 1
        cfg, env = self.cfg, self.env
        position, q = env.start, self.dfa.start
        trace: List[str] = []
        last_event_cell = None
        steps = 0
        for steps in range(1, env.max_steps + 1):
            state = (position, q)
            a = self._act(state, greedy=not learn)
            position, event, terminal = env.step(position, env.actions[a], last_event_cell)
            q_next = self.dfa.step(q, event)
            if event is not None:
                trace.append(event)
                last_event_cell = position
            self.visits[q_next] += 1
            accepted = self.dfa.is_accepting(q_next)
            done = terminal or accepted
            if learn:
                r = self._reward(q, q_next) + self._bonus(q_next)
                target = r if done else r + cfg.gamma * self.q[(position, q_next)].max()
                self.q[state][a] += cfg.alpha * (target - self.q[state][a])
            q = q_next
            if done:
                break

        w = tuple(trace)
        outcome = self.task.reward(w)          # the only feedback about the task: R(w) in {0, 1}
        refined = False
        if learn:
            self.epsilon = max(cfg.epsilon_min, self.epsilon * cfg.epsilon_decay)
            if cfg.refine:
                inconsistent = self.dfa.accepts(w) != bool(outcome)
                gap = cfg.coverage_gap and any(self.confidence.get(s, 0.0) < cfg.theta_min
                                               for s in run_states(self.dfa, w))
                if inconsistent or gap:
                    self._refine(w, outcome)
                    refined = True
            self.log.success.append(outcome)
            self.log.steps.append(steps)
            self.log.refined.append(int(refined))
            self.log.dfa_states.append(self.dfa.num_states)
            self.log.patterns_known.append(len(self.patterns_known()))
            self.log.wall_time.append(time.perf_counter() - self._start)
            self.log.samples.append(sum(self.log.steps))
        return outcome, w
