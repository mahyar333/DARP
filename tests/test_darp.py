import numpy as np
import pytest

from darp.agent import DARPAgent, DARPConfig
from darp.automata import DFA, rpni
from darp.baselines.agi import AGIConfig, run_agi
from darp.confidence import confidence_scores
from darp.envs import make
from darp.product import transfer_q_values

DQ_NEG = [("dragon",), ("chest", "dragon"), ("shield", "dragon")]


# ---------------------------------------------------------------- tasks & envs
def test_dungeon_quest_reward_and_patterns():
    _, task = make("dungeon_quest")
    assert task.reward(("key", "chest", "shield", "dragon")) == 1
    assert task.reward(("chest", "key", "shield", "dragon")) == 0      # chest before key: no sword
    assert task.reward(("chest", "key", "chest", "shield", "dragon")) == 1
    assert task.pattern(("shield", "key", "chest", "dragon")) == "shield-key-chest"
    assert all(task.reward(w) for w in task.patterns.values())


def test_blind_craftsman_reward_and_patterns():
    _, task = make("blind_craftsman")
    assert task.reward(("wood", "wood", "wood", "factory", "wood", "factory", "home")) == 1  # 3rd wood is wasted
    assert task.reward(("wood", "factory", "wood", "factory", "home")) == 0                  # only 2 tools
    assert task.pattern(("wood", "wood", "factory", "wood", "factory", "home")) == "2+1"
    assert all(task.reward(w) for w in task.patterns.values())


def test_event_rule():
    env, _ = make("blind_craftsman")
    pos, event, _ = env.step((0, 1), "down")                # enter wood at (1, 1)
    assert pos == (1, 1) and event == "wood"
    pos, event, _ = env.step((1, 1), "up", last_event_cell=(1, 1))
    pos, event, _ = env.step(pos, "down", last_event_cell=(1, 1))
    assert event is None                                     # re-entering the same source emits nothing
    assert env.step((1, 1), "up")[0] == (0, 1)
    assert env.step((2, 1), "right")[0] == (2, 1)            # obstacle at (2, 2)


# ---------------------------------------------------------------- automata
def test_confidence_formula():
    dfa = DFA(2, {(0, "a"): 1}, accepting={1})
    c = confidence_scores(dfa, positives=[("a",)], negatives=[(), ("b",)], lam=0.4, beta=4.0)
    assert c[0] == pytest.approx((1 + 0.4 * 2) / (3 + 4))   # q0: 1 positive, 2 negatives
    assert c[1] == pytest.approx(1 / (1 + 4))                # q1: 1 positive


def test_rpni_is_consistent():
    _, task = make("dungeon_quest")
    rng = np.random.default_rng(0)
    alphabet = ["key", "chest", "shield", "dragon"]
    traces = {tuple(rng.choice(alphabet, rng.integers(0, 7))) for _ in range(300)}
    P = [w for w in traces if task.reward(w)]
    N = [w for w in traces if not task.reward(w)]
    dfa = rpni(P, N)
    assert all(dfa.accepts(w) for w in P) and not any(dfa.accepts(w) for w in N)


def test_q_transfer_keeps_unchanged_states_and_seeds_changed_ones():
    old = DFA(2, {(0, "a"): 1}, accepting={1})
    new = DFA(3, {(0, "a"): 1, (1, "b"): 2}, accepting={2})   # q1 gains an edge and stops accepting
    q_old = {((0, 0), 0): np.array([1.0, 2.0]), ((0, 0), 1): np.array([3.0, 4.0])}
    q_new = transfer_q_values(q_old, old, new, [(0, 0)], 2, eps_init=0.25)
    assert np.allclose(q_new[((0, 0), 0)], [1.0, 2.0])        # unchanged
    assert np.allclose(q_new[((0, 0), 1)], [0.25, 0.25])      # changed -> eps_init
    assert np.allclose(q_new[((0, 0), 2)], [2.0, 3.0])        # new -> average over old states


# ---------------------------------------------------------------- agents
def test_inconsistency_triggers_refinement():
    env, task = make("dungeon_quest")
    agent = DARPAgent(env, task, DARPConfig(coverage_gap=False), [], DQ_NEG)
    assert not agent.dfa.accepting
    agent._refine(task.patterns["key-chest-shield"], 1)
    assert agent.dfa.accepts(task.patterns["key-chest-shield"])
    assert "key-chest-shield" in agent.patterns_known()


def test_passive_baseline_never_refines():
    env, task = make("dungeon_quest")
    agent = DARPAgent(env, task, DARPConfig(refine=False, exploration_bonus=False), [], DQ_NEG)
    for _ in range(20):
        agent.run_episode()
    assert agent.refinements == 0 and agent.dfa.num_states == 1


def test_darp_learns_dungeon_quest():
    env, task = make("dungeon_quest")
    cfg = DARPConfig(alpha=0.6, gamma=0.9, epsilon_min=0.03, epsilon_decay=0.985, progress_reward=0.4, step_penalty=0.02)
    agent = DARPAgent(env, task, cfg, [], DQ_NEG, seed=0)
    for _ in range(300):
        agent.run_episode()
    assert np.mean(agent.log.success[-20:]) >= 0.9
    assert len(agent.patterns_known()) >= 1


def test_agi_baseline_runs():
    env, task = make("dungeon_quest")
    _, log = run_agi(env, task, AGIConfig(episodes=5, episode_length=50, alpha=0.4, gamma=0.85,
                                          epsilon_decay=0.99, vi_gamma=0.85), seed=0)
    assert len(log.success) == 5 and log.queries[-1] > 0
