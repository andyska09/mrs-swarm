"""Replay buffer. One per species, conspecific experience merged."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    obs: jnp.ndarray  # (capacity, obs_dim)
    action: jnp.ndarray  # (capacity, act_dim)
    reward: jnp.ndarray  # (capacity,)
    next_obs: jnp.ndarray  # (capacity, obs_dim)
    ptr: jnp.ndarray  # ()
    size: jnp.ndarray  # ()


def empty(capacity, obs_dim, act_dim):
    return Buffer(
        obs=jnp.zeros((capacity, obs_dim)),
        action=jnp.zeros((capacity, act_dim)),
        reward=jnp.zeros((capacity,)),
        next_obs=jnp.zeros((capacity, obs_dim)),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


def insert(buf, obs, action, reward, next_obs):
    """One row per conspecific, written in a single call."""
    capacity, n = buf.obs.shape[0], obs.shape[0]
    idx = (buf.ptr + jnp.arange(n)) % capacity
    return buf._replace(
        obs=buf.obs.at[idx].set(obs),
        action=buf.action.at[idx].set(action),
        reward=buf.reward.at[idx].set(reward),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        ptr=((buf.ptr + n) % capacity).astype(jnp.int32),
        size=jnp.minimum(buf.size + n, capacity).astype(jnp.int32),
    )


def sample(buf, key, batch_size):
    """Uniform over the filled region. -> (obs, action, reward, next_obs)."""
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buf.size, 1))
    return buf.obs[idx], buf.action[idx], buf.reward[idx], buf.next_obs[idx]
