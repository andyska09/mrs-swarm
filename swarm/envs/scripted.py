"""Scripted policies — fixed rules, no learning.

The predator rule is the paper's own (section 4.4): rotate the heading to point
at the nearest prey, then move toward it at maximum speed. It is the gate-2
baseline a learned predator has to beat, the prey's opponent in the swirling
experiment, and a pure-physics exercise of the environment.
"""
import jax
import jax.numpy as jnp

from swarm.envs.dynamics import wrap_angle
from swarm.envs.predator_prey import pairwise


def scripted_predator(state, params):
    """-> (n_pred, 2) actions in [-1, 1]. Full throttle, turn toward nearest prey."""
    n0 = params.n_pred
    rel, dist = pairwise(state.pos, params)
    target = jnp.argmin(dist[:n0, n0:], axis=-1)                  # nearest prey per predator
    to_prey = jnp.take_along_axis(rel[:n0, n0:], target[:, None, None], axis=1)[:, 0]

    err = wrap_angle(jnp.arctan2(to_prey[:, 1], to_prey[:, 0]) - state.theta[:n0])
    # a_R is an angular velocity: the turn we want this step, capped by the limit.
    a_r = jnp.clip(err / (params.dt * params.max_ang_vel), -1.0, 1.0)
    return jnp.stack([jnp.ones(n0), a_r], axis=-1)


def random_actions(key, n):
    return jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)


def scripted_vs_random(key, obs, state, params):
    """Scripted predators, random prey. The phase-1 gate-1 policy."""
    prey = random_actions(key, params.n_prey)
    if params.n_pred == 0:
        return prey
    return jnp.concatenate([scripted_predator(state, params), prey], axis=0)


def all_random(key, obs, state, params):
    return random_actions(key, params.n_pred + params.n_prey)
