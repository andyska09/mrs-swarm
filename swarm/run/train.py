"""Train and write everything needed to plot, render, or evaluate later.

    python -m swarm.run.train --preset flocking --exp exp_flocking
    python -m swarm.run.train --preset pendulum --exp exp_skeleton --seeds 0

Seeds run as one vmapped program, then each is written to its own leaf
runs/<exp>/<preset>/s<seed>/ holding config.json, params.pkl, metrics.npz,
summary.json and results.png. Leaves are no-clobber without --overwrite, so a
path is fixed by exp, preset and seed alone and aggregation can find it.
"""
import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm.algo.config import PRESETS, get_train_config
from swarm.run.plot import plot_metrics

ROOT = Path(__file__).resolve().parents[2]


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="flocking", choices=list(PRESETS))
    ap.add_argument("--exp", default=None, help="experiment folder; default exp_<preset>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=None, help="override the preset")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def build(cfg):
    """The only place an env id becomes an env. -> (train_fn, description)."""
    if cfg.env_id == "predator_prey":
        from swarm.algo.train_swarm import make_train
        from swarm.envs import predator_prey as pp
        params = pp.get_env_params(cfg.env_preset)
        return make_train(cfg, params), f"predator_prey/{cfg.env_preset} " \
                                        f"({params.n_pred}v{params.n_prey})"
    import gymnax
    from swarm.algo.train_gymnax import make_train
    env, env_params = gymnax.make(cfg.env_id)
    return make_train(cfg, env, env_params), cfg.env_id


def save(out, cfg, config, result, i, wall, n_seeds):
    take = lambda tree: jax.device_get(jax.tree.map(lambda x: x[i], tree))
    metrics = {k: np.asarray(v[i]) for k, v in result["metrics"].items()}
    payload = {"config": config, **{k: take(result[k]) for k in result if k != "metrics"}}
    with open(out / "params.pkl", "wb") as f:
        pickle.dump(payload, f)
    np.savez(out / "metrics.npz", **metrics)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tail = lambda k, n=100: float(np.nanmean(metrics[k][-n:]))
    head = lambda k, n=100: float(np.nanmean(metrics[k][:n]))
    summary = {**{k: config[k] for k in ("preset", "exp", "seed")},
               "total_steps": cfg.total_steps, "wall_s": round(wall, 1),
               "steps_per_s": int(cfg.total_steps * n_seeds / wall),
               "device": str(jax.devices()[0]),
               **{f"{k}_first": head(k) for k in metrics},
               **{f"{k}_final": tail(k) for k in metrics}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_metrics(metrics, out / "results.png", title=f"{config['exp']}/{config['preset']}/s{i}")
    return summary


def main():
    a = parse()
    exp = a.exp or f"exp_{a.preset}"
    cfg = get_train_config(a.preset)
    if a.episodes:
        cfg = replace(cfg, episodes=a.episodes)

    leaves = [ROOT / "runs" / exp / a.preset / f"s{s}" for s in a.seeds]
    for out in leaves:
        if out.exists() and any(out.iterdir()) and not a.overwrite:
            raise SystemExit(f"{out} exists and is not empty. Use --overwrite.")
        out.mkdir(parents=True, exist_ok=True)

    train, desc = build(cfg)
    print(f"device={jax.devices()[0]}  env={desc}  preset={a.preset}  seeds={a.seeds}  "
          f"{cfg.episodes} x {cfg.episode_len} = {cfg.total_steps} steps each")

    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    t0 = time.time()
    result = jax.block_until_ready(jax.jit(jax.vmap(train))(keys))
    wall = time.time() - t0

    for i, (seed, out) in enumerate(zip(a.seeds, leaves)):
        config = {**asdict(replace(cfg, seed=seed)), "preset": a.preset, "exp": exp}
        s = save(out, cfg, config, result, i, wall, len(a.seeds))
        key = "dos" if "dos_final" in s else "ep_return"
        print(f"  s{seed}: {key} {s[key + '_first']:.3f} -> {s[key + '_final']:.3f}  -> {out}")
    print(f"done in {wall:.0f}s ({int(cfg.total_steps * len(a.seeds) / wall)} steps/s total)")


if __name__ == "__main__":
    main()
