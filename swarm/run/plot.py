"""Training curves from metrics.npz.

    python -m swarm.run.plot runs/exp_flocking/flocking/s0
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(x, w):
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_metrics(metrics, path, title=""):
    keys = [k for k in metrics if metrics[k].ndim == 1]
    ncol = 3
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    w = max(1, len(metrics[keys[0]]) // 50)

    for ax, k in zip(axes.flat, keys):
        y = metrics[k]
        ax.plot(y, alpha=0.25, lw=0.8)
        s = smooth(y, w)
        ax.plot(np.arange(len(s)) + w // 2, s, lw=1.6)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    m = dict(np.load(a.run_dir / "metrics.npz"))
    plot_metrics(m, a.run_dir / "results.png", title=str(a.run_dir))
    print(f"-> {a.run_dir / 'results.png'}")


if __name__ == "__main__":
    main()
