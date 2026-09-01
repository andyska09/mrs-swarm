"""Statistics over many episodes of one eval config.

    python -m swarm_simple.run.eval eval_configs/flock50.json --episodes 200

-> evals/<name>.json. Same episode as run.replay, vmapped over env seeds.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm_simple.config import eval_as_dict, load_eval
from swarm_simple.envs import metrics
from swarm_simple.envs import predator_prey as pp
from swarm_simple.run.replay import episode_fn

ROOT = Path(__file__).resolve().parents[2]


def _stat(x):
    """-> mean over everything, and sd of the per-episode means."""
    x = np.asarray(x)
    return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(np.nanmean(x, axis=1)))}


def evaluate(cfg, episodes):
    env, n0 = cfg.env, cfg.env.n_pred
    keys = jnp.stack([jax.random.PRNGKey(cfg.env_seed + i) for i in range(episodes)])
    states, reward, info = jax.jit(jax.vmap(episode_fn(cfg)))(keys)

    per_step = jax.vmap(jax.vmap(lambda p, t: (metrics.dos(p, env), metrics.doa(p, t, env))))
    dos, doa = per_step(states.pos[:, :, n0:], states.theta[:, :, n0:])
    tail = slice(3 * env.episode_len // 4, None)

    return {
        "episodes": episodes,
        "captures_per_step": _stat(info["captures"]),
        "captures_per_step_per_prey": _stat(info["captures"] / env.n_prey),
        "prey_return": _stat(reward[:, :, n0:].sum(axis=1)),
        "pred_return": _stat(reward[:, :, :n0].sum(axis=1)) if n0 else None,
        "dos": _stat(dos),
        "doa": _stat(doa),
        "dos_final_quarter": _stat(dos[:, tail]),
        "doa_final_quarter": _stat(doa[:, tail]),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    cfg = load_eval(a.config)
    out = evaluate(cfg, a.episodes)

    path = ROOT / "evals" / f"{cfg.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": eval_as_dict(cfg), **out}, indent=2))

    print(f"{cfg.name}  {cfg.env.n_pred}v{cfg.env.n_prey}  {a.episodes} episodes")
    for k in ("captures_per_step", "dos", "dos_final_quarter", "doa", "doa_final_quarter"):
        print(f"  {k:<20}{out[k]['mean']:8.4f} +- {out[k]['sd']:.4f}")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
