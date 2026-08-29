"""Create a run from a config JSON.

python -m swarm_simple.run.train configs/flocking.json --seeds 0 1 2
"""

import argparse
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm_simple.config import as_dict, config_hash, load

ROOT = Path(__file__).resolve().parents[2]


def make_train(cfg):
    """The only place an algorithm name turns into code."""
    if cfg.algo == "maddpg":
        from swarm_simple.algo.maddpg import make_train as build

        return build(cfg)
    raise SystemExit(f"no builder for algo {cfg.algo!r}")


def commit():
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def make_run(cfg, seeds):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = config_hash(cfg)
    run = ROOT / "runs" / f"{stamp}_{digest}"
    for s in seeds:
        (run / f"s{s}").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(as_dict(cfg), indent=2))
    (run / "meta.json").write_text(
        json.dumps(
            {
                "created": stamp,
                "config_hash": digest,
                "commit": commit(),
                "seeds": list(seeds),
            },
            indent=2,
        )
    )
    return run


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    cfg = load(a.config)
    run = make_run(cfg, a.seeds)
    print(
        f"{cfg.name}  {cfg.algo}  {cfg.env.n_pred}v{cfg.env.n_prey}  "
        f"{cfg.train.episodes} x {cfg.env.episode_len} steps  "
        f"seeds={a.seeds}  {jax.devices()[0]}"
    )

    keys = jnp.stack([jax.random.PRNGKey(s) for s in a.seeds])
    t0 = time.time()
    result = jax.block_until_ready(jax.jit(jax.vmap(make_train(cfg)))(keys))
    wall = time.time() - t0

    metrics = result.pop("metrics")
    for i, seed in enumerate(a.seeds):
        leaf = run / f"s{seed}"
        np.savez(leaf / "metrics.npz", **{k: np.asarray(v[i]) for k, v in metrics.items()})
        with open(leaf / "params.pkl", "wb") as f:
            pickle.dump(jax.device_get(jax.tree.map(lambda x: x[i], result)), f)

    meta = json.loads((run / "meta.json").read_text())
    (run / "meta.json").write_text(json.dumps({**meta, "wall_s": round(wall, 1)}, indent=2))
    print(f"-> {run.relative_to(ROOT)}  in {wall:.0f}s")


if __name__ == "__main__":
    main()
