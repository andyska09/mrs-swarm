"""Train and write everything needed to plot, render, or re-evaluate later.

    python -m swarm.run.train --preset pendulum --exp exp_skeleton --seed 0

Writes runs/<exp>/<preset>/s<seed>/:
    config.json    full config + preset name, flat, so the tree is greppable
    params.pkl     network params + obs-norm stats + a copy of the config
    metrics.npz    per-episode arrays
    summary.json   final numbers, wall time, steps/s, device
    results.png    training curves

Paths are deterministic and the leaf is no-clobber (--overwrite to replace), so
aggregation can locate a run from its preset, exp and seed alone.
"""
import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from swarm.algo.config import PRESETS, get_train_config
from swarm.run.plot import plot_metrics

ROOT = Path(__file__).resolve().parents[2]


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="pendulum", choices=list(PRESETS))
    ap.add_argument("--exp", default="exp_skeleton", help="experiment folder; groups an ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None, help="override the preset")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def build(cfg):
    """-> (train_fn, description). The only place an env id becomes an env."""
    if cfg.env_id == "predator_prey":
        raise NotImplementedError("phase 2: the two-species loop is not written yet")
    import gymnax
    from swarm.algo.train_gymnax import make_train
    env, env_params = gymnax.make(cfg.env_id)
    return make_train(cfg, env, env_params), cfg.env_id


def main():
    a = parse()
    cfg = get_train_config(a.preset)
    cfg = replace(cfg, seed=a.seed, **({"episodes": a.episodes} if a.episodes else {}))

    out = ROOT / "runs" / a.exp / a.preset / f"s{a.seed}"
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} exists and is not empty. Use --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    train, desc = build(cfg)
    print(f"device={jax.devices()[0]}  env={desc}  preset={a.preset}  seed={cfg.seed}  "
          f"{cfg.episodes} x {cfg.episode_len} = {cfg.total_steps} steps")

    t0 = time.time()
    result = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(cfg.seed)))
    wall = time.time() - t0

    metrics = {k: np.asarray(v) for k, v in result["metrics"].items()}
    norm = result["obs_norm"]
    config = {**asdict(cfg), "preset": a.preset, "exp": a.exp}

    with open(out / "params.pkl", "wb") as f:
        # Obs-norm stats are PART OF THE POLICY; the config is duplicated here so
        # a checkpoint stays self-contained.
        pickle.dump({"agent": jax.device_get(result["agent"]),
                     "obs_mean": np.asarray(norm.mean), "obs_var": np.asarray(norm.var),
                     "config": config}, f)
    np.savez(out / "metrics.npz", **metrics)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tail = lambda k, n=20: float(np.nanmean(metrics[k][-n:]))
    summary = {"preset": a.preset, "exp": a.exp, "seed": cfg.seed,
               "total_steps": cfg.total_steps, "wall_s": round(wall, 1),
               "steps_per_s": int(cfg.total_steps / wall), "device": str(jax.devices()[0]),
               "first_return": float(np.nanmean(metrics["ep_return"][:20])),
               "final_return": tail("ep_return")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_metrics(metrics, out / "results.png", title=f"{a.exp}/{a.preset}/s{a.seed}")

    print(f"done in {wall:.0f}s ({summary['steps_per_s']} steps/s) | "
          f"return {summary['first_return']:.1f} -> {summary['final_return']:.1f} | -> {out}")


if __name__ == "__main__":
    main()
