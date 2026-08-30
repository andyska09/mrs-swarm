"""Run one episode from an eval config.

    python -m swarm_simple.run.replay eval_configs/flock50.json

-> renders/<name>/{traj.npz, config.json}. Draw it with run.render.

The eval env is free to differ from the one a policy trained in — population,
arena, physics, rewards. The actor is shared within a species and sees a fixed
(n, d_o), so only the observation width is fixed by the checkpoint.
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm_simple.algo import networks
from swarm_simple.config import eval_as_dict, load, load_eval
from swarm_simple.envs import predator_prey as pp
from swarm_simple.envs import scripted

ROOT = Path(__file__).resolve().parents[2]


def _run_config(sp, env_cfg):
    cfg = load(ROOT / sp.run / "config.json")
    # Dense_0 is (obs_dim, hidden[0]); a mismatch is a shape error deep inside apply.
    if pp.obs_dim(cfg.env) != pp.obs_dim(env_cfg):
        raise SystemExit(
            f"{sp.run}: obs_dim {pp.obs_dim(cfg.env)} != {pp.obs_dim(env_cfg)}. "
            "n_neighbors and heading_encoding must match the run."
        )
    return cfg


def actor_for(sp, species, env_cfg, key):
    """-> f(obs) -> (n, 2) in [-1, 1], or None for the scripted rule."""
    if sp.mode == "scripted":
        return None
    if sp.mode == "random":
        return lambda k, obs: jax.random.uniform(
            k, (obs.shape[0], 2), minval=-1.0, maxval=1.0
        )

    cfg = _run_config(sp, env_cfg)
    net = networks.build(cfg.model, pp.act_dim(env_cfg))[0]
    if sp.mode == "untrained":
        params = net.init(key, jnp.zeros((1, pp.obs_dim(env_cfg))))
    else:
        with open(ROOT / sp.run / f"s{sp.seed}" / "params.pkl", "rb") as f:
            params = pickle.load(f)[f"{species}_actor"]
    # Bare mu(o): eps and noise are a learning device, not part of the policy.
    return lambda k, obs: net.apply(params, obs)[0]


def simulate(cfg):
    """-> (states, reward, info). Not `pp.rollout`: scripted reads the state, not the obs."""
    env_cfg, n0 = cfg.env, cfg.env.n_pred
    pred = actor_for(cfg.pred, "pred", env_cfg, jax.random.PRNGKey(cfg.pred.seed))
    prey = actor_for(cfg.prey, "prey", env_cfg, jax.random.PRNGKey(cfg.prey.seed))

    def tick(carry, _):
        state, key = carry
        key, k_pred, k_prey = jax.random.split(key, 3)
        obs = pp.observe(state, env_cfg)
        a_pred = (
            scripted.predator(state, env_cfg)
            if pred is None
            else pred(k_pred, obs[:n0])
        )
        action = jnp.concatenate([a_pred, prey(k_prey, obs[n0:])])
        state = pp.step(state, action, env_cfg)
        r, info = pp.reward(state, action, env_cfg)
        return (state, key), (state, r, info)

    def go(key):
        _, out = jax.lax.scan(
            tick, (pp.reset(key, env_cfg), key), None, length=env_cfg.episode_len
        )
        return out

    return jax.jit(go)(jax.random.PRNGKey(cfg.env_seed))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    a = ap.parse_args()

    cfg = load_eval(a.config)
    states, reward, info = simulate(cfg)

    out = ROOT / "renders" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "traj.npz",
        pos=np.asarray(states.pos),
        vel=np.asarray(states.vel),
        theta=np.asarray(states.theta),
    )
    (out / "config.json").write_text(json.dumps(eval_as_dict(cfg), indent=2))

    n0 = cfg.env.n_pred
    print(
        f"{cfg.name}  {n0}v{cfg.env.n_prey}  pred={cfg.pred.mode} prey={cfg.prey.mode}\n"
        f"captures={float(info['captures'].sum()):.0f}  "
        f"prey return={float(reward[:, n0:].sum(0).mean()):.2f}\n"
        f"-> {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
