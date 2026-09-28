"""Run DARP and baselines on a benchmark.

    python -m darp.run --config configs/dungeon_quest.yaml
    python -m darp.run --config configs/blind_craftsman.yaml --methods darp hipo --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from .agent import DARPAgent, DARPConfig, EpisodeLog
from .baselines.agi import AGIConfig, run_agi
from .envs import make

METHODS = ("darp", "static", "hipo", "agi")
NAMES = {"darp": "DARP", "static": "Static RPNI", "hipo": "HiPO", "agi": "AGI (L*)"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def darp_config(cfg: dict, method: str) -> DARPConfig:
    values = dict(cfg["q_learning"])
    if method == "darp":
        values.update({k: v for k, v in cfg["darp"].items() if k != "start_with_known_pattern"})
    else:  # passive baselines: no refinement and no exploration bonus
        values.update(refine=False, exploration_bonus=False)
    names = {f.name for f in fields(DARPConfig)}
    return DARPConfig(**{k: v for k, v in values.items() if k in names})


def run_method(method: str, cfg: dict, seed: int) -> EpisodeLog:
    env, task = make(cfg["env"])
    if method == "agi":
        _, log = run_agi(env, task, AGIConfig(episodes=cfg["episodes"], **cfg["agi"]), seed)
        return log

    knowledge = cfg["initial_knowledge"]
    negatives = [tuple(s.split()) for s in knowledge["negatives"]]
    known = [task.patterns[knowledge["known_pattern"]]]
    if method == "darp":
        positives = known if cfg["darp"].get("start_with_known_pattern", False) else []
    elif method == "hipo":
        positives = known
    else:
        positives = []
    agent = DARPAgent(env, task, darp_config(cfg, method), positives, negatives, seed)
    for _ in range(cfg["episodes"]):
        agent.run_episode()
    return agent.log


def summarize(log: EpisodeLog, window: int = 20, threshold: float = 0.9, tail: int = 50) -> dict:
    s = np.asarray(log.success, dtype=float)
    avg = np.convolve(s, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(avg >= threshold)
    first = int(hit[0] + window - 1) if len(hit) else None
    ok_steps = [st for st, ok in zip(log.steps[-tail:], log.success[-tail:]) if ok]
    return {
        # Runs that never reach 90% count at their full budget (a lower bound).
        "reached_90": first is not None,
        "episodes_to_90": first + 1 if first is not None else len(s),
        "samples_to_90": int(log.samples[first] if first is not None else log.samples[-1]),
        "final_success": float(s[-tail:].mean()),
        "new_patterns": int(log.patterns_known[-1]),
        "refinements": int(sum(log.refined)),
        "steps_when_successful": float(np.mean(ok_steps)) if ok_steps else None,
        "queries": int(log.queries[-1]) if log.queries else 0,
        "wall_time_s": float(log.wall_time[-1]),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    out = Path(args.out or Path("results") / cfg["env"])
    out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for method in args.methods:
        runs = []
        for seed in seeds:
            log = run_method(method, cfg, seed)
            (out / f"{method}_seed{seed}.json").write_text(json.dumps(asdict(log)))
            runs.append(summarize(log))
        summaries[method] = runs
        print_row(method, runs, cfg["episodes"])
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nLogs saved to {out}/  (plot with: python -m darp.plot {out})")


def aggregate(runs, key, default=None):
    vals = [r[key] if r[key] is not None else default for r in runs]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def print_row(method, runs, episodes):
    conv = aggregate(runs, "episodes_to_90")
    reached = sum(r["reached_90"] for r in runs)
    fin = aggregate(runs, "final_success")
    pat = aggregate(runs, "new_patterns")
    q = aggregate(runs, "queries")
    smp = aggregate(runs, "samples_to_90")
    print(f"{NAMES[method]:12s} episodes to 90%: {conv[0]:6.1f} ± {conv[1]:5.1f} | "
          f"env. samples to 90%: {'>=' if reached < len(runs) else '  '}{smp[0]:8.0f} ± {smp[1]:6.0f} "
          f"({reached}/{len(runs)} runs) | "
          f"final success: {100 * fin[0]:5.1f} ± {100 * fin[1]:4.1f}% | "
          f"new patterns: {pat[0]:.1f} | oracle queries: {q[0]:.0f}", flush=True)


if __name__ == "__main__":
    main()
