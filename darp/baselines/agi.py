"""AGI baseline: Active Grammatical Inference for non-Markovian planning.

Re-implementation of the L*-based approach of Topper et al., "Active
Grammatical Inference for Non-Markovian Planning" (ICAPS 2022), adapted from
the reference implementation to run on the environments of this package:

* an L* learner builds an observation table from membership queries and
  proposes a hypothesis DFA;
* equivalence is checked on the model up to depth ``omega`` and, during
  learning, by counterexamples found while executing the policy;
* the policy is initialised by value iteration on M x H and refined with
  Q-learning for reward machines (QRM) with counterfactual updates;
* every counterexample triggers another round of L*.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, product
from math import inf
from typing import Dict, Optional, Tuple

from ..agent import EpisodeLog
from ..envs import LabeledGridWorld, Task

EMPTY = ""  # label of semantically empty states


# --------------------------------------------------------------------------
# Environment interface expected by the algorithm
# --------------------------------------------------------------------------
class AGIEnv:
    """Exposes a grid world as a known MDP with a hidden reward machine.

    The transition model P and labelling L are available to the learner; the
    reward is only observed through queries and interaction. Events are
    emitted when a labelled cell is entered, as in the DARP environments.
    """

    def __init__(self, env: LabeledGridWorld, task: Task):
        self.env = env
        self.task = task
        self.actions = list(env.actions)
        self.states = list(env.states)
        self.state_0 = env.start
        self.labels = tuple(sorted(env.alphabet))  # fixed order: set order of strings varies between runs
        self.reset()

    def L(self, state) -> str:
        return self.env.label(state) or EMPTY

    def P(self, state, action) -> Dict[Tuple[int, int], float]:
        return {self.env.move(state, action): 1.0}

    def reset(self):
        self.state = self.state_0
        self.hidden = self.task.initial()
        self.finished = False
        self.last_event_cell = None
        return self.state

    def step(self, action):
        """Returns (state, event label or EMPTY, reward, episode finished)."""
        nxt, event, _ = self.env.step(self.state, action, self.last_event_cell)
        self.state = nxt
        if event is not None:
            self.last_event_cell = nxt
        if event is None or self.finished:
            return nxt, EMPTY, 0, self.finished
        self.hidden, reward, self.finished = self.task.update(self.hidden, event)
        return nxt, event, reward, self.finished

    def trajectory_reward(self, trajectory) -> int:
        """Reward observed at the last state of a trajectory of labelled states."""
        hidden = self.task.initial()
        reward = 0
        for i, state in enumerate(trajectory):
            hidden, reward, finished = self.task.update(hidden, self.L(state))
            if finished:
                return reward if i == len(trajectory) - 1 else 0
        return 0


# --------------------------------------------------------------------------
# Hypothesis automaton
# --------------------------------------------------------------------------
class HypothesisDFA:
    def __init__(self, alphabet, states, accepting, state_0, trans):
        self.alphabet = alphabet
        self.states = states
        self.accepting = accepting
        self.state_0 = state_0
        self.trans = trans

    def transition(self, q, a):
        if (q, a) in self.trans:
            return self.trans[(q, a)]
        if a == EMPTY:
            return q
        return None


# --------------------------------------------------------------------------
# L*: observation table, teacher and learner
# --------------------------------------------------------------------------
class ObservationTable:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.S = {()}
        self.E = {()}
        self.T = {}

    def closed(self):
        """Return (True, None) or (False, long trace without a matching row in S)."""
        for long_trace in self.long_traces():
            if not any(self.row_eq(s, long_trace) for s in self.S):
                return False, long_trace
        return True, None

    def consistent(self):
        """Return (True, None) or (False, suffix that separates two equal rows)."""
        for s1, s2 in combinations(self.S, 2):
            if not self.row_eq(s1, s2):
                continue
            for a, e in product(self.alphabet, self.E):
                if self.T[s1 + (a,) + e] != self.T[s2 + (a,) + e]:
                    return False, (a,) + e
        return True, None

    def fill(self, teacher):
        for s, e in product(self.S | self.long_traces(), self.E):
            if s + e not in self.T:
                self.T[s + e] = teacher.membership(s + e)

    def long_traces(self):
        return {s + (a,) for s, a in product(self.S, self.alphabet)}

    def row(self, s):
        return tuple(self.T[s + e] for e in self.E)

    def row_eq(self, s1, s2):
        return self.row(s1) == self.row(s2)


class Teacher:
    """Answers queries using the known MDP and the observed reward."""

    def __init__(self, mdp: AGIEnv):
        self.mdp = mdp
        self.samples = 0      # number of queries
        self.env_steps = 0    # environment steps needed to execute the queried trajectories

    def membership(self, query):
        self.samples += 1
        state, trajectory = self.mdp.state_0, ()
        for label in query:
            state, steps = self._bfs(state, label)
            self.env_steps += steps
            trajectory = trajectory + (state,)
        return self.mdp.trajectory_reward(trajectory)

    def equivalence(self, H: HypothesisDFA, max_depth: int):
        """Search M x H breadth-first for a trace on which H and M disagree."""
        queue = deque([(self.mdp.state_0, H.state_0, (), 0)])
        while queue:
            mdp_state, dfa_state, trajectory, cost = queue.popleft()
            if len(trajectory) > max_depth:
                break
            self.samples += 1
            self.env_steps += cost
            reward = self.mdp.trajectory_reward(trajectory)
            if int(dfa_state in H.accepting) != reward:
                return self.trace(trajectory)
            for label in H.alphabet:
                next_mdp_state, steps = self._bfs(mdp_state, label)
                queue.append((next_mdp_state, H.transition(dfa_state, label),
                              trajectory + (next_mdp_state,), cost + steps))
        return None

    def mdp_bfs(self, source, label):
        """Nearest state labelled ``label`` reachable through unlabelled states."""
        return self._bfs(source, label)[0]

    def _bfs(self, source, label):
        """Like :meth:`mdp_bfs`, also returning the number of steps to reach the state."""
        queue = deque((s, 1) for s in self.possible_next_states(source, label))
        visited = {s for s, _ in queue}
        while queue:
            state, dist = queue.popleft()
            if self.mdp.L(state) == label:
                return state, dist
            for next_state in self.possible_next_states(state, label):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, dist + 1))
        return None, 0

    def possible_next_states(self, state, label):
        next_states = set()
        for action in self.mdp.actions:
            for next_state, prob in self.mdp.P(state, action).items():
                if prob > 0 and self.mdp.L(next_state) in (label, EMPTY):
                    next_states.add(next_state)
        return next_states

    def trace(self, trajectory):
        return tuple(self.mdp.L(s) for s in trajectory if self.mdp.L(s) != EMPTY)


class Learner:
    def __init__(self, alphabet, omega: int):
        self.alphabet = alphabet
        self.omega = omega
        self.H: Optional[HypothesisDFA] = None

    def hypothesis(self, table: ObservationTable) -> HypothesisDFA:
        states = {table.row(s) for s in table.S}
        accepting = {table.row(s) for s in table.S if table.T[s] == 1}
        trans = {(table.row(s), a): table.row(s + (a,)) for s, a in product(table.S, self.alphabet)}
        return HypothesisDFA(self.alphabet, states, accepting, table.row(()), trans)

    def learn(self, table: ObservationTable, teacher: Teacher, counterexample=None) -> None:
        table.fill(teacher)
        while True:
            if counterexample is not None:
                for i in range(len(counterexample)):
                    table.S |= {counterexample[: i + 1]}
                table.fill(teacher)

            while True:
                closed, s = table.closed()
                consistent, e = table.consistent()
                if not consistent:
                    table.E |= {e}
                elif not closed:
                    table.S |= {s}
                else:
                    break
                table.fill(teacher)

            H = self.hypothesis(table)
            counterexample = teacher.equivalence(H, self.omega)
            if counterexample is None:
                self.H = H
                return


# --------------------------------------------------------------------------
# Planning: value iteration warm start + QRM
# --------------------------------------------------------------------------
class ValueIteration:
    def __init__(self, mdp: AGIEnv, dfa: HypothesisDFA, gamma: float, theta: float = 1e-4):
        self.mdp = mdp
        self.dfa = dfa
        self.gamma = gamma
        self.theta = theta
        self.Q = defaultdict(float)

    def iterate(self) -> None:
        delta = inf
        while delta >= self.theta:
            delta = 0.0
            for s, q, a in product(self.mdp.states, self.dfa.states, self.mdp.actions):
                old = self.Q[s, q, a]
                new = 0.0
                for s1, prob in self.mdp.P(s, a).items():
                    label = self.mdp.L(s1)
                    q1 = self.dfa.transition(q, label)
                    r = 1 if q1 in self.dfa.accepting and label != EMPTY else 0
                    new += prob * (r + self.gamma * max(self.Q[s1, q1, a1] for a1 in self.mdp.actions))
                self.Q[s, q, a] = new
                delta = max(delta, abs(old - new))


class QRM:
    """Q-learning for reward machines with counterfactual updates."""

    def __init__(self, mdp: AGIEnv, dfa: HypothesisDFA, episode_length: int,
                 alpha: float, gamma: float, epsilon_decay: float, rng: random.Random):
        self.mdp = mdp
        self.dfa = dfa
        self.episode_length = episode_length
        self.actions = mdp.actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.rng = rng
        self.Q = defaultdict(float)

    def _bellman(self, s, q, a, r, s1, q1) -> None:
        target = r + self.gamma * max(self.Q[s1, q1, a1] for a1 in self.actions)
        self.Q[s, q, a] = (1 - self.alpha) * self.Q[s, q, a] + self.alpha * target

    def episode(self):
        """Run one episode. Returns (counterexample, reward) or (None, None)."""
        trace, rewards = (), ()
        s, q = self.mdp.reset(), self.dfa.state_0
        steps = 0
        for _ in range(self.episode_length):
            steps += 1
            a = self.epsilon_greedy(s, q)
            s1, label, reward, finished = self.mdp.step(a)
            q1 = self.dfa.transition(q, label)
            if label != EMPTY:
                trace += (label,)
                rewards += (reward,)

            reward_hyp = 1 if q1 in self.dfa.accepting and label != EMPTY else 0
            if reward != reward_hyp:
                return (trace, reward), sum(rewards), steps

            self._bellman(s, q, a, reward, s1, q1)
            # Counterfactual updates for every other automaton state.
            for q_other in self.dfa.states - {q}:
                q_other1 = self.dfa.transition(q_other, label)
                r_other = 1 if q_other1 in self.dfa.accepting and label != EMPTY else 0
                self._bellman(s, q_other, a, r_other, s1, q_other1)
            s, q = s1, q1
            if finished:
                break
        return None, sum(rewards), steps

    def epsilon_greedy(self, s, q):
        if self.rng.random() < self.epsilon:
            a = self.rng.choice(self.actions)
        else:
            best = max(self.Q[s, q, a] for a in self.actions)
            a = self.rng.choice([a for a in self.actions if self.Q[s, q, a] == best])
        self.epsilon *= self.epsilon_decay
        return a


@dataclass
class AGIConfig:
    episodes: int
    episode_length: int
    alpha: float
    gamma: float
    epsilon_decay: float
    vi_gamma: float   # discount used by the value-iteration warm start
    omega: int = 5    # depth of the model-based equivalence check


def hypothesis_accepts(H: HypothesisDFA, trace) -> bool:
    q = H.state_0
    for label in trace:
        q = H.transition(q, label)
        if q is None:
            return False
    return q in H.accepting


def run_agi(env: LabeledGridWorld, task: Task, cfg: AGIConfig, seed: int,
            log_every: int = 0) -> Tuple[HypothesisDFA, EpisodeLog]:
    """Train the AGI baseline. Reported time includes automaton learning."""
    rng = random.Random(seed)
    mdp = AGIEnv(env, task)
    log = EpisodeLog()
    start = time.perf_counter()

    teacher = Teacher(mdp)
    table = ObservationTable(mdp.labels)
    learner = Learner(mdp.labels, cfg.omega)

    def relearn(counterexample=None):
        learner.learn(table, teacher, counterexample)
        vi = ValueIteration(mdp, learner.H, cfg.vi_gamma)
        vi.iterate()
        agent = QRM(mdp, learner.H, cfg.episode_length, cfg.alpha, cfg.gamma, cfg.epsilon_decay, rng)
        agent.Q = vi.Q
        return agent

    agent = relearn()
    for episode in range(cfg.episodes):
        counterexample, reward, steps = agent.episode()
        if counterexample is not None:
            agent = relearn(counterexample[0])
        H = learner.H
        log.success.append(int(reward > 0))
        log.steps.append(steps)
        log.refined.append(int(counterexample is not None))
        log.dfa_states.append(len(H.states))
        log.patterns_known.append(sum(hypothesis_accepts(H, w) for w in task.patterns.values()))
        log.wall_time.append(time.perf_counter() - start)
        log.queries.append(teacher.samples)
        log.samples.append(sum(log.steps) + teacher.env_steps)
        if log_every and episode % log_every == 0:
            print(f"episode {episode:5d} | reward {reward} | hypothesis states {len(H.states)}")
    return learner.H, log
