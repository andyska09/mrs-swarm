"""DoS and DoA

The paper's claim is that swarming emerges while these go unoptimised.

Both are defined over the nearest CONSPECIFIC.
"""

import jax.numpy as jnp

from swarm_simple.envs.predator_prey import delta, max_distance


def nearest(pos, cfg):
    """-> (index of the nearest conspecific, distance to it), per agent."""
    d = jnp.linalg.norm(delta(pos[:, None, :], pos[None, :, :], cfg), axis=-1)
    d = d.at[jnp.diag_indices(pos.shape[0])].set(jnp.inf)
    return jnp.argmin(d, axis=-1), jnp.min(d, axis=-1)


def dos(pos, cfg):
    """Mean nearest-neighbour distance over D. Eq 2.

    Not density-normalised, so it only compares across runs with the same
    population: nearest-neighbour distance falls as 1/sqrt(N).
    """
    return jnp.mean(nearest(pos, cfg)[1]) / max_distance(cfg)


def doa(pos, theta, cfg):
    """Mean ||h_j + h_k|| / 2 over each agent and its nearest conspecific. Eq 3.

    h is always the true unit heading, whatever heading_encoding the observation
    uses — eq 3 is defined on the physics, not on the network input.
    """
    k = nearest(pos, cfg)[0]
    h = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return jnp.mean(jnp.linalg.norm(h + h[k], axis=-1)) / 2.0
