"""One episode -> GIF. Scatter + quiver, one frame per step.

    python -m swarm.run.render --preset torus --seed 0

A debugging tool from phase 1 on. DoS = 0.31 is the same number for flocking,
orbiting and stuck in a corner; the GIF tells them apart at a glance.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.envs import metrics, predator_prey as pp
from swarm.envs.scripted import scripted_vs_random

ROOT = Path(__file__).resolve().parents[2]


def frames(traj, params, stride=1):
    n0, half = params.n_pred, params.edge / 2.0
    pos, theta = np.asarray(traj.pos), np.asarray(traj.theta)
    periodic = params.boundary == "torus"

    for t in range(0, pos.shape[0], stride):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=110)
        p, th = pos[t], theta[t]
        u, v = np.cos(th), np.sin(th)

        ax.quiver(p[:, 0], p[:, 1], u, v, color="0.6", width=0.004,
                  scale=22, headwidth=3, zorder=1)
        ax.scatter(p[n0:, 0], p[n0:, 1], s=26, c="tab:blue", zorder=2)
        ax.scatter(p[:n0, 0], p[:n0, 1], s=90, c="tab:orange", zorder=3)

        prey = jax.numpy.asarray(p[n0:])
        d = metrics.dos(prey, params.edge, periodic)
        a = metrics.doa(prey, jax.numpy.asarray(th[n0:]), params.edge, periodic)
        ax.set_title(f"t={t:3d}   DoS={d:.3f}  DoA={a:.3f}", fontsize=10, family="monospace")

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        fig.canvas.draw()
        yield np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="torus", choices=list(pp.PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    params = pp.get_env_params(a.preset)
    traj, rewards, info = jax.jit(pp.rollout, static_argnums=(1, 2))(
        jax.random.PRNGKey(a.seed), params, scripted_vs_random)

    out = a.out or ROOT / "runs" / "exp_physics" / a.preset / f"s{a.seed}" / "render.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, list(frames(traj, params, a.stride)), fps=a.fps, loop=0)
    print(f"captures/step {float(np.mean(info['captures'])):.2f} | "
          f"prey return {float(np.asarray(rewards)[:, params.n_pred:].sum(0).mean()):.2f} | -> {out}")


if __name__ == "__main__":
    main()
