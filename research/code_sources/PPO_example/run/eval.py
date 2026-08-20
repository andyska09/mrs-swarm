"""Deterministic evaluation of a saved policy (mean action, no exploration noise).

    python run/eval.py runs/straight_s0                 # eval on the training preset
    python run/eval.py runs/straight_s0 --preset weave   # transfer test
    python run/eval.py runs/straight_s0 --episodes 2000

The policy was trained behind obs normalization; the running mean/var are
saved in params.pkl and applied here. Forgetting this is the classic
"trained fine, evals as random" bug.
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from envs import Interceptor2D, get_env_params, PRESETS
from ppo.config import TrainConfig
from ppo.train import build_network


def load(run_dir):
    with open(Path(run_dir) / "params.pkl", "rb") as f:
        d = pickle.load(f)
    cfg = TrainConfig(**d["train_config"])
    net = build_network(cfg, d["action_dim"])
    return d, net, cfg


def make_policy(net, params, obs_mean, obs_var):
    mean, var = jnp.asarray(obs_mean), jnp.asarray(obs_var)

    def act(obs):
        pi, _ = net.apply(params, (obs - mean) / jnp.sqrt(var + 1e-8))
        return pi.mode()
    return act


def evaluate(policy, env, env_params, rng, episodes):
    def _episode(rng):
        obs, st = env.reset(rng, env_params)

        def _step(c, _):
            obs, st, done, ret, cap, length, rng = c
            rng, k = jax.random.split(rng)
            obs, st, r, d, info = env.step(k, st, policy(obs), env_params)
            alive = 1.0 - done.astype(jnp.float32)
            ret = ret + r * alive
            cap = cap + (info["r_capture"] > 0).astype(jnp.float32) * alive
            length = length + alive
            return (obs, st, done | d, ret, cap, length, rng), None

        init = (obs, st, jnp.bool_(False), jnp.float32(0), jnp.float32(0), jnp.float32(0), rng)
        (_, _, _, ret, cap, length, _), _ = jax.lax.scan(
            _step, init, None, length=env_params.max_steps_in_episode)
        return ret, cap, length

    rets, caps, lens = jax.jit(jax.vmap(_episode))(jax.random.split(rng, episodes))
    return {"mean_return": float(rets.mean()), "std_return": float(rets.std()),
            "capture_rate": float(caps.mean()), "mean_length": float(lens.mean()),
            "episodes": int(episodes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--preset", default=None, choices=list(PRESETS))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    d, net, cfg = load(a.run_dir)
    preset = a.preset or d["preset"]
    env, env_params = Interceptor2D(), get_env_params(preset)
    policy = make_policy(net, d["params"], d["obs_mean"], d["obs_var"])
    res = evaluate(policy, env, env_params, jax.random.PRNGKey(a.seed), a.episodes)
    print(f"[{Path(a.run_dir).name} on {preset}]  " +
          "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))


if __name__ == "__main__":
    main()
