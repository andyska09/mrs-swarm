"""Deterministic rollout of a trained policy.

    python -m swarm.run.eval runs/exp_flocking/flocking/s0
    python -m swarm.run.eval runs/exp_flocking/flocking/s0 --preset eval50 --gif

Actions are greedy and the parameters stay frozen. The env preset is free to
differ from the one the policy trained on — the paper trains on 10 prey and
deploys on 50.
"""
import argparse
import json
import pickle
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm.algo import ddpg
from swarm.algo.config import TrainConfig
from swarm.envs import metrics, predator_prey as pp


def load(run_dir):
    with open(Path(run_dir) / "params.pkl", "rb") as f:
        payload = pickle.load(f)
    if "pred" not in payload:
        raise SystemExit(f"{run_dir} is not a predator-prey run")
    names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in payload["config"].items() if k in names})
    return payload, cfg


def make_policy(payload, cfg, params):
    algo = ddpg.make_ddpg(cfg, pp.obs_dim(params), 2)
    n0 = params.n_pred

    def policy(key, obs, state, _params):
        flat = pp.flatten_obs(obs)
        k0, k1 = jax.random.split(key)
        a0 = algo.act(payload["pred"], ddpg.normalize(payload["pred_norm"], flat[:n0]), k0, 0.0, 0.0)
        a1 = algo.act(payload["prey"], ddpg.normalize(payload["prey_norm"], flat[n0:]), k1, 0.0, 0.0)
        return jnp.concatenate([a0, a1])

    return policy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--preset", default=None, help="env preset; default the one it trained on")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="override episode length")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    payload, cfg = load(a.run_dir)
    preset = a.preset or cfg.env_preset
    params = pp.get_env_params(preset)
    if a.steps:
        params = params.replace(episode_len=a.steps)
    policy = make_policy(payload, cfg, params)
    periodic = params.boundary == "torus"

    keys = jnp.stack([jax.random.PRNGKey(1000 + i) for i in range(a.episodes)])
    traj, rewards, info = jax.jit(jax.vmap(lambda k: pp.rollout(k, params, policy)))(keys)

    prey_pos, prey_th = traj.pos[:, :, params.n_pred:], traj.theta[:, :, params.n_pred:]
    per_step = jax.vmap(jax.vmap(lambda p, t: (metrics.dos(p, params.edge, periodic),
                                               metrics.doa(p, t, params.edge, periodic))))
    dos, doa = per_step(prey_pos, prey_th)

    out = {"run": str(a.run_dir), "env_preset": preset, "episodes": a.episodes,
           "dos": float(dos.mean()), "dos_std": float(dos.mean(1).std()),
           "doa": float(doa.mean()), "doa_std": float(doa.mean(1).std()),
           "dos_final_quarter": float(dos[:, -params.episode_len // 4:].mean()),
           "doa_final_quarter": float(doa[:, -params.episode_len // 4:].mean()),
           "captures_per_step": float(info["captures"].mean()),
           "prey_return": float(np.asarray(rewards)[:, :, params.n_pred:].sum(1).mean())}
    (a.run_dir / f"eval_{preset}.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"  {k:20s} {v}")

    if a.gif:
        import imageio.v2 as imageio
        from swarm.run.render import frames
        ep0 = jax.tree.map(lambda x: x[0], traj)
        path = a.run_dir / f"eval_{preset}.gif"
        imageio.mimsave(path, list(frames(ep0, params)), fps=15, loop=0)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
