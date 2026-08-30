"""
A replayed episode -> GIF.

    python -m swarm_simple.run.render renders/flock50

Reads run.replay output.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from swarm_simple.config import load_eval
from swarm_simple.envs import metrics
from swarm_simple.envs import predator_prey as pp

ARROW_FRAC = 0.06  # arrow length at max speed, as a fraction of the arena edge


def _discs(ax, pos, r, color, zorder):
    """units='xy' draws a body at its true radius in metres, not in points."""
    ax.add_collection(
        EllipseCollection(
            2 * r,
            2 * r,
            0,
            units="xy",
            offsets=pos,
            offset_transform=ax.transData,
            facecolors=color,
            zorder=zorder,
        )
    )


def frame(pos, vel, theta, cfg):
    n0, half = cfg.n_pred, cfg.edge / 2
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)

    r = np.asarray(pp.radii(cfg))
    _discs(ax, pos[n0:], r[n0:], "tab:blue", 2)
    _discs(ax, pos[:n0], r[:n0], "tab:orange", 3)

    speed = np.linalg.norm(vel, axis=-1) / np.asarray(pp.max_speeds(cfg))
    length = speed * ARROW_FRAC * cfg.edge
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        np.cos(theta) * length,
        np.sin(theta) * length,
        color="0.35",
        width=0.004,
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        zorder=4,
    )

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def frames(traj, cfg):
    pos, vel, theta = traj["pos"], traj["vel"], traj["theta"]
    n0 = cfg.n_pred
    for t in range(pos.shape[0]):
        fig, ax = frame(pos[t], vel[t], theta[t], cfg)
        title = f"t={t:3d}"
        if cfg.n_prey >= 2:
            d = metrics.dos(jnp.asarray(pos[t, n0:]), cfg)
            a = metrics.doa(jnp.asarray(pos[t, n0:]), jnp.asarray(theta[t, n0:]), cfg)
            title += f"   DoS={d:.3f}  DoA={a:.3f}"
        ax.set_title(title, fontsize=10, family="monospace")
        fig.tight_layout()
        fig.canvas.draw()
        yield np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir", type=Path, help="a renders/<name>/ written by run.replay")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = load_eval(a.dir / "config.json").env
    traj = np.load(a.dir / "traj.npz")

    out = a.out or a.dir / "out.gif"
    imageio.mimsave(out, list(frames(traj, cfg)), fps=a.fps, loop=0)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
