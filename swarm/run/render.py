"""Episodes -> GIF. Scatter + quiver, one frame per step, one panel per policy.

    python -m swarm.run.render --preset torus --seed 0
    python -m swarm.run.render --compare runs/exp_radius/rad10/s0 --seed 7

A debugging tool from phase 1 on. DoS = 0.31 is the same number for flocking,
orbiting and stuck in a corner; the GIF tells them apart at a glance.
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

from swarm.algo import ddpg
from swarm.envs import metrics, predator_prey as pp
from swarm.envs.scripted import scripted_vs_random

ROOT = Path(__file__).resolve().parents[2]


def _trails(ax, hist, params):
    """hist (t, N, 2). Torus wraps are broken with NaN so paths do not streak."""
    n0, xs, ys = params.n_pred, hist[..., 0].copy(), hist[..., 1].copy()
    wrap = np.abs(np.diff(hist, axis=0)).max(-1) > params.edge / 2.0
    xs[1:][wrap] = np.nan
    ys[1:][wrap] = np.nan
    ax.plot(xs[:, n0:], ys[:, n0:], lw=0.7, alpha=0.45, color="tab:blue", zorder=0)
    ax.plot(xs[:, :n0], ys[:, :n0], lw=1.1, alpha=0.6, color="tab:orange", zorder=0)


def _discs(ax, pos, r, color, zorder):
    """units='xy' draws the body at its true radius in metres, not in points."""
    ax.add_collection(EllipseCollection(2 * r, 2 * r, 0, units="xy", offsets=pos,
                                        offset_transform=ax.transData,
                                        facecolors=color, zorder=zorder))


def _panel(ax, hist, theta, params, label, t, trail=0):
    n0, half = params.n_pred, params.edge / 2.0
    pos = hist[-1]
    if trail:
        _trails(ax, hist[-trail:], params)
    # Arrows are 2.6 body radii long and drawn over the discs, so the heading
    # stays visible when the radius presets inflate the bodies.
    r = np.asarray(pp.radii(params))
    _discs(ax, pos[n0:], r[n0:], "tab:blue", 2)
    _discs(ax, pos[:n0], r[:n0], "tab:orange", 3)
    ax.quiver(pos[:, 0], pos[:, 1], np.cos(theta) * 2.6 * r, np.sin(theta) * 2.6 * r,
              color="0.35", width=0.004, angles="xy", scale_units="xy", scale=1,
              headwidth=3, zorder=4)

    d = metrics.dos(jnp.asarray(pos[n0:]), params.edge, params.boundary == "torus")
    a = metrics.doa(jnp.asarray(pos[n0:]), jnp.asarray(theta[n0:]),
                    params.edge, params.boundary == "torus")
    head = f"{label}\n" if label else ""
    ax.set_title(f"{head}t={t:3d}   DoS={d:.3f}  DoA={a:.3f}", fontsize=10, family="monospace")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def frames(trajs, params, stride=1, labels=None, trail=0):
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    labels = labels or [""] * len(trajs)
    pos = [np.asarray(t.pos) for t in trajs]
    theta = [np.asarray(t.theta) for t in trajs]

    for t in range(0, pos[0].shape[0], stride):
        fig, axes = plt.subplots(1, len(trajs), figsize=(4.5 * len(trajs), 4.9),
                                 dpi=110, squeeze=False)
        for ax, p, th, lab in zip(axes.flat, pos, theta, labels):
            _panel(ax, p[:t + 1], th[t], params, lab, t, trail)
        fig.tight_layout()
        fig.canvas.draw()
        yield np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)


def untrained_like(cfg, params, key):
    """A freshly initialised policy — the paper's 'before coevolution'."""
    from swarm.run.eval import make_policy
    algo = ddpg.make_ddpg(cfg, pp.obs_dim(params), 2)
    k0, k1 = jax.random.split(key)
    dim = pp.obs_dim(params)
    return make_policy({"pred": algo.init(k0), "prey": algo.init(k1),
                        "pred_norm": ddpg.new_obsnorm(dim),
                        "prey_norm": ddpg.new_obsnorm(dim)}, cfg, params)


def _length(params, steps):
    return params.replace(episode_len=steps) if steps else params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, choices=list(pp.PRESETS))
    ap.add_argument("--compare", type=Path, default=None, help="run dir: untrained beside trained")
    ap.add_argument("--runs", nargs="+", type=Path, default=None,
                    help="run dirs: one trained panel each, labelled by cell")
    ap.add_argument("--untrained", action="store_true", help="prepend an untrained panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--trail", type=int, default=0, help="draw the last N steps of each path")
    ap.add_argument("--steps", type=int, default=None, help="episode length; overrides the preset")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    roll = lambda p, pol: jax.jit(pp.rollout, static_argnums=(1, 2))(
        jax.random.PRNGKey(a.seed), p, pol)

    if a.runs:
        from swarm.run.eval import load, make_policy
        loaded = [load(r) for r in a.runs]
        params = _length(pp.get_env_params(a.preset or loaded[0][1].env_preset), a.steps)
        trajs, labels = [], []
        if a.untrained:
            trajs.append(roll(params, untrained_like(loaded[0][1], params, jax.random.PRNGKey(0)))[0])
            labels.append("untrained")
        for r, (payload, cfg) in zip(a.runs, loaded):
            trajs.append(roll(params, make_policy(payload, cfg, params))[0])
            labels.append(r.parent.name)
        out = a.out or a.runs[0].parents[1] / "compare.gif"
    elif a.compare:
        from swarm.run.eval import load, make_policy
        payload, cfg = load(a.compare)
        params = _length(pp.get_env_params(a.preset or cfg.env_preset), a.steps)
        # Same env seed both sides, so the initial state is identical.
        trajs = [roll(params, untrained_like(cfg, params, jax.random.PRNGKey(0)))[0],
                 roll(params, make_policy(payload, cfg, params))[0]]
        labels = ["before coevolution", "after coevolution"]
        out = a.out or a.compare / "before_after.gif"
    else:
        params = _length(pp.get_env_params(a.preset or "torus"), a.steps)
        trajs, labels = [roll(params, scripted_vs_random)[0]], ["scripted predators, random prey"]
        out = a.out or ROOT / "runs" / "exp_physics" / params.boundary / f"s{a.seed}" / "render.gif"

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, list(frames(trajs, params, a.stride, labels, a.trail)),
                fps=a.fps, loop=0)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
