"""Predator-prey environment. Spec section 1.

Agents are an array predators first: `pos[:n_pred]`, then `pos[n_pred:]`.

`cfg` is an EnvConfig and is passed to jit as a static argument.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

SPAWN_TRIES = 100  # bound on the rejection sampler; dense arenas give up and overlap


class State(NamedTuple):
    pos: jnp.ndarray  # (N, 2)
    vel: jnp.ndarray  # (N, 2)
    theta: jnp.ndarray  # (N,)
    time: jnp.ndarray  # ()


def n_agents(cfg):
    return cfg.n_pred + cfg.n_prey


def _per_agent(cfg, pred, prey):
    return jnp.concatenate([jnp.full(cfg.n_pred, pred), jnp.full(cfg.n_prey, prey)])


def radii(cfg):
    return _per_agent(cfg, cfg.radius_pred, cfg.radius_prey)


def masses(cfg):
    return _per_agent(cfg, cfg.mass_pred, cfg.mass_prey)


def max_speeds(cfg):
    return _per_agent(cfg, cfg.speed_pred, cfg.speed_prey)


def max_distance(cfg):
    """D in eq 2: the largest separation two agents can have."""
    return cfg.edge * 2**0.5 / (2.0 if cfg.boundary == "torus" else 1.0)


def perception(cfg):
    return cfg.perception_frac * max_distance(cfg)


def delta(a, b, cfg):
    """b - a, under the minimum-image convention on a torus."""
    d = b - a
    if cfg.boundary == "torus":
        d = (d + cfg.edge / 2) % cfg.edge - cfg.edge / 2
    return d


def _spawn_pos(key, cfg):
    n, half, r = n_agents(cfg), cfg.edge / 2, radii(cfg)
    unplaced = jnp.arange(n)

    def place(i, carry):
        pos, key = carry

        def clear(p):
            d = jnp.linalg.norm(delta(pos, p, cfg), axis=-1)
            return jnp.all((d >= r + r[i]) | (unplaced >= i))

        def cond(c):
            tries, _, p = c
            return (tries < SPAWN_TRIES) & ~clear(p)

        def body(c):
            tries, key, _ = c
            key, k = jax.random.split(key)
            return (
                tries + 1,
                key,
                jax.random.uniform(k, (2,), minval=-half, maxval=half),
            )

        key, k = jax.random.split(key)
        p = jax.random.uniform(k, (2,), minval=-half, maxval=half)
        _, key, p = jax.lax.while_loop(cond, body, (0, key, p))
        return pos.at[i].set(p), key

    pos, _ = jax.lax.fori_loop(0, n, place, (jnp.zeros((n, 2)), key))
    return pos


def _spawn_vel(key, cfg):
    n = n_agents(cfg)
    k_dir, k_mag = jax.random.split(key)
    ang = jax.random.uniform(k_dir, (n,), minval=-jnp.pi, maxval=jnp.pi)
    mag = jax.random.uniform(k_mag, (n,)) * cfg.init_speed_frac * max_speeds(cfg)
    return mag[:, None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)


def reset(key, cfg):
    k_pos, k_vel, k_theta = jax.random.split(key, 3)
    return State(
        pos=_spawn_pos(k_pos, cfg),
        vel=_spawn_vel(k_vel, cfg),
        theta=jax.random.uniform(
            k_theta, (n_agents(cfg),), minval=-jnp.pi, maxval=jnp.pi
        ),
        time=jnp.int32(0),
    )
