"""Plot learning curves saved by ``python -m darp.run``.

    python -m darp.plot results/dungeon_quest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"darp": "#2a78d6", "hipo": "#1baf7a", "static": "#eb6834", "agi": "#eda100"}
NAMES = {"darp": "DARP", "hipo": "HiPO", "static": "Static RPNI", "agi": "AGI (L*, uses oracle)"}
ORDER = ["darp", "hipo", "static", "agi"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(results_dir: Path):
    logs = {}
    for path in sorted(Path(results_dir).glob("*_seed*.json")):
        method = path.stem.split("_seed")[0]
        logs.setdefault(method, []).append(json.loads(path.read_text()))
    return logs


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:window] / np.arange(1, window)
    return np.concatenate([head, (c[window:] - c[:-window]) / window])


def plot_success(logs, path: Path, title: str = "", window: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for method in ORDER:
        if method not in logs:
            continue
        curves = np.stack([moving_average(run["success"], window) for run in logs[method]])
        mean, std = curves.mean(0), curves.std(0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, color=COLORS[method], lw=2, label=NAMES[method],
                ls="--" if method == "agi" else "-")
        ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=COLORS[method], alpha=0.12, lw=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate ({window}-episode average)")
    ax.set_ylim(0, 1.03)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9.5, handlelength=1.8)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args(argv)
    out = Path(args.results_dir)
    logs = load(out)
    if not logs:
        raise SystemExit(f"No *_seed*.json logs in {out}")
    (out / "figures").mkdir(exist_ok=True)
    plot_success(logs, out / "figures" / "success_rate.png", out.name.replace("_", " ").title())
    print(f"Figures written to {out / 'figures'}/")


if __name__ == "__main__":
    main()
