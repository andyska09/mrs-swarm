"""The paper's §4.4 predator rule: turn toward the nearest prey, full throttle.

Replaces the learned predator so that only the prey evolve.
"""

import jax.numpy as jnp

from swarm_simple.envs.predator_prey import delta, wrap_angle


def predator(state, cfg):
    """-> (n_pred, 2) actions in [-1, 1]."""
    n0 = cfg.n_pred
    rel = delta(state.pos[:n0, None, :], state.pos[None, n0:, :], cfg)
    dist = jnp.linalg.norm(rel, axis=-1)
    nearest = jnp.argmin(dist, axis=-1)[:, None, None]
    to_prey = jnp.take_along_axis(rel, nearest, axis=1)[:, 0]

    err = wrap_angle(jnp.arctan2(to_prey[:, 1], to_prey[:, 0]) - state.theta[:n0])
    # a_R is an angular velocity: the turn wanted this step, capped by the limit.
    a_r = jnp.clip(err / (cfg.dt * cfg.max_ang_vel), -1.0, 1.0)
    return jnp.stack([jnp.ones(n0), a_r], axis=-1)
