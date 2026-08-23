"""Combine seeds into one number per cell and one figure per experiment.

    python -m swarm.run.aggregate exp_npredators

Writes runs/<exp>/<preset>/aggregate.json, runs/<exp>/comparison.png, and the
tracked table results/<exp>.md.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CURVES = ("dos", "doa", "captures", "prey_reward")
WINDOW = 100        # the paper's running-average length


def running_mean(x, w):
    if x.shape[-1] < w:
        return x
    k = np.ones(w) / w
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), -1, x)


def load_cell(cell_dir):
    seeds = sorted(cell_dir.glob("s*/metrics.npz"))
    if not seeds:
        return None
    per_seed = [dict(np.load(p)) for p in seeds]
    return {k: np.stack([m[k] for m in per_seed]) for k in per_seed[0]}, len(seeds)


def ci95(x, axis=0):
    n = x.shape[axis]
    return 1.96 * np.nanstd(x, axis=axis) / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    a = ap.parse_args()
    exp_dir = ROOT / "runs" / a.exp
    cells = {}

    for cell_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        loaded = load_cell(cell_dir)
        if loaded is None:
            continue
        m, n = loaded
        cells[cell_dir.name] = m
        tail = {k: (float(np.nanmean(v[:, -WINDOW:])), float(ci95(np.nanmean(v[:, -WINDOW:], 1))))
                for k, v in m.items()}
        head = {k: float(np.nanmean(v[:, :WINDOW])) for k, v in m.items()}
        (cell_dir / "aggregate.json").write_text(json.dumps(
            {"seeds": n, "window": WINDOW,
             "first": head, "final": {k: v[0] for k, v in tail.items()},
             "final_ci95": {k: v[1] for k, v in tail.items()}}, indent=2))

    if not cells:
        raise SystemExit(f"no runs under {exp_dir}")

    keys = [k for k in CURVES if k in next(iter(cells.values()))]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        for name, m in cells.items():
            s = running_mean(m[k], WINDOW)
            mean, band = np.nanmean(s, 0), ci95(s)
            x = np.arange(mean.size) + WINDOW // 2
            ax.plot(x, mean, lw=1.6, label=name)
            ax.fill_between(x, mean - band, mean + band, alpha=0.2)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(a.exp)
    fig.tight_layout()
    fig.savefig(exp_dir / "comparison.png", dpi=120)
    plt.close(fig)

    rows = ["| cell | seeds | DoS first → final | DoA first → final | captures/step |",
            "|---|---|---|---|---|"]
    for name, m in cells.items():
        agg = json.loads((exp_dir / name / "aggregate.json").read_text())
        f, c, h = agg["final"], agg["final_ci95"], agg["first"]
        rows.append(f"| {name} | {agg['seeds']} | "
                    f"{h['dos']:.3f} → {f['dos']:.3f} ± {c['dos']:.3f} | "
                    f"{h['doa']:.3f} → {f['doa']:.3f} ± {c['doa']:.3f} | "
                    f"{f['captures']:.3f} ± {c['captures']:.3f} |")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / f"{a.exp}.md").write_text(
        f"# {a.exp}\n\nMean over the last {WINDOW} episodes, ± 95% CI across seeds.\n\n"
        + "\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\n-> {exp_dir / 'comparison.png'}  and  results/{a.exp}.md")


if __name__ == "__main__":
    main()
