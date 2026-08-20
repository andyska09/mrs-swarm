"""Training curve + a few deterministic trajectories -> <run_dir>/results.png

    python run/plot.py runs/straight_s0
    python run/plot.py runs/straight_s0 --episodes 6 --preset weave
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import Interceptor2D, get_env_params, PRESETS
from run.eval import load, make_policy


def trajectory(policy, env, env_params, rng):
    # step_env, NOT step: gymnax's env.step auto-resets on done, so the state
    # it returns at episode end is a FRESH episode (pursuer back at origin).
    # Logging that is the classic "every trajectory ends at (0,0)" artefact.
    obs, st = env.reset(rng, env_params)
    step = jax.jit(env.step_env)
    pp, pt = [np.asarray(st.pos_p)], [np.asarray(st.pos_t)]
    for _ in range(env_params.max_steps_in_episode):
        rng, k = jax.random.split(rng)
        obs, st, r, done, info = step(k, st, policy(obs), env_params)
        pp.append(np.asarray(st.pos_p)); pt.append(np.asarray(st.pos_t))
        if bool(done):
            break
    return np.array(pp), np.array(pt), bool(info["r_capture"] > 0)


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1): i + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--preset", default=None, choices=list(PRESETS))
    ap.add_argument("--episodes", type=int, default=5)
    a = ap.parse_args()
    run = Path(a.run_dir)

    m = np.load(run / "metrics.npz")
    d, net, cfg = load(run)
    preset = a.preset or d["preset"]
    env, env_params = Interceptor2D(), get_env_params(preset)
    policy = make_policy(net, d["params"], d["obs_mean"], d["obs_var"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(m["mean_return"])) * cfg.num_envs * cfg.num_steps / 1e6

    ax[0].plot(x, m["mean_return"], lw=0.5, alpha=0.3)
    ax[0].plot(x, smooth(m["mean_return"]), lw=2, label="mean return (smoothed)")
    ax[0].set_xlabel("env steps [M]"); ax[0].set_title(f"return — {run.name}"); ax[0].grid(alpha=.3); ax[0].legend()

    if "capture_rate" in m:
        ax[1].plot(x, smooth(m["capture_rate"]), lw=2, label="capture rate")
        ax[1].plot(x, smooth(m["miss_rate"]), lw=2, label="miss rate")
        ax[1].set_ylim(-0.02, 1.02); ax[1].legend()
    ax[1].set_xlabel("env steps [M]"); ax[1].set_title("outcomes"); ax[1].grid(alpha=.3)

    n_cap = 0
    for i in range(a.episodes):
        pp, pt, cap = trajectory(policy, env, env_params, jax.random.PRNGKey(100 + i))
        n_cap += cap
        c = f"C{i}"
        ax[2].plot(pp[:, 0], pp[:, 1], color=c, lw=1.5)
        ax[2].plot(pt[:, 0], pt[:, 1], color=c, lw=1.5, ls="--", alpha=.7)
        ax[2].plot(*pp[0], "o", color=c, ms=6); ax[2].plot(*pt[0], "s", color=c, ms=6)
        ax[2].plot(*pp[-1], "*" if cap else "x", color=c, ms=12)
    ax[2].plot([], [], "k-", label="pursuer"); ax[2].plot([], [], "k--", label="target")
    ax[2].plot([], [], "k*", label="capture"); ax[2].plot([], [], "kx", label="no capture")
    ax[2].set_aspect("equal"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    ax[2].set_title(f"{preset}: {n_cap}/{a.episodes} captured (deterministic)")

    plt.tight_layout()
    outp = run / "results.png"
    plt.savefig(outp, dpi=130)
    print(f"saved {outp}  ({n_cap}/{a.episodes} captured)")


if __name__ == "__main__":
    main()
