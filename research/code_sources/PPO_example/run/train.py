"""Train PPO on Interceptor2D and save everything needed to eval/plot later.

    python run/train.py                                   # defaults: straight, 50M steps
    python run/train.py --preset weave --steps 100e6 --seed 3 --out runs/weave_s3
    python run/train.py --steps 2e6 --num-envs 256        # CPU smoke run (~1 min)

Writes to --out:
    params.pkl     trained network params + the two configs (enough to rebuild the net)
    metrics.npz    one array per metric, shape (num_updates,)
    summary.json   final numbers, throughput, device
"""
import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from envs import Interceptor2D, get_env_params, PRESETS
from ppo.config import TrainConfig
from ppo.train import make_train


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="straight", choices=list(PRESETS))
    ap.add_argument("--steps", type=float, default=50e6, help="total env steps")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--num-steps", type=int, default=128, help="rollout length")
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="default: runs/<preset>_s<seed>")
    return ap.parse_args()


def main():
    a = parse()
    out = Path(a.out) if a.out else ROOT / "runs" / f"{a.preset}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(num_envs=a.num_envs, num_steps=a.num_steps, num_minibatches=a.num_minibatches,
                      total_timesteps=int(a.steps), seed=a.seed, lr=a.lr, gamma=a.gamma,
                      ent_coef=a.ent_coef)
    env = Interceptor2D()
    env_params = get_env_params(a.preset)

    print(f"device={jax.devices()[0]}  preset={a.preset}  steps={cfg.total_timesteps/1e6:.1f}M  "
          f"envs={cfg.num_envs}  updates={cfg.num_updates}  seed={cfg.seed}")
    train = jax.jit(make_train(cfg, env, env_params))

    t0 = time.time()
    outp = train(jax.random.PRNGKey(cfg.seed))
    jax.block_until_ready(outp)
    wall = time.time() - t0
    sps = cfg.total_timesteps / wall

    metrics = {k: np.asarray(v) for k, v in outp["metrics"].items()}
    params = jax.device_get(outp["runner_state"][0].params)
    # The policy was trained behind NormalizeVecObservation: its running
    # mean/var are PART OF THE POLICY. Peel the wrapper states to get them.
    # Chain (outermost first): NormalizeVecReward -> NormalizeVecObservation -> ...
    es = outp["runner_state"][1]
    if cfg.normalize_reward:
        es = es.env_state
    obs_mean = np.asarray(es.mean) if cfg.normalize_obs else np.zeros(env.obs_size, np.float32)
    obs_var = np.asarray(es.var) if cfg.normalize_obs else np.ones(env.obs_size, np.float32)
    with open(out / "params.pkl", "wb") as f:
        pickle.dump({"params": params, "obs_mean": obs_mean, "obs_var": obs_var,
                     "train_config": asdict(cfg), "preset": a.preset,
                     "obs_size": env.obs_size, "action_dim": env.num_actions}, f)
    np.savez(out / "metrics.npz", **metrics)

    def tail(k, n=10):
        v = metrics[k][-n:]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    summary = {
        "preset": a.preset, "seed": cfg.seed, "total_timesteps": cfg.total_timesteps,
        "num_updates": int(cfg.num_updates), "wall_s": round(wall, 1),
        "steps_per_s": int(sps), "device": str(jax.devices()[0]),
        "final_mean_return": tail("mean_return"),
        "final_capture_rate": tail("capture_rate") if "capture_rate" in metrics else None,
        "final_ep_length": tail("mean_ep_length"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {wall:.0f}s ({sps/1e3:.0f}k steps/s) | return {summary['final_mean_return']:.2f}"
          f" | capture {summary['final_capture_rate']:.2f} | saved -> {out}")


if __name__ == "__main__":
    main()
