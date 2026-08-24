"""DoS and DoA (Li 2023 eqs 2-3). Pure functions of positions and headings.

Diagnostics only. The paper's claim is that swarming emerges while these go
unoptimised, so they live here, take bare arrays, and stay out of the reward.

Both are defined over the NEAREST CONSPECIFIC. Reference value: uniformly random
headings give DoA = E[cos(phi/2)] = 2/pi ~ 0.637.
"""
import jax.numpy as jnp


def nearest_neighbor(pos, edge, periodic):
    """-> (index of nearest conspecific, distance to it), per agent."""
    rel = pos[None, :, :] - pos[:, None, :]
    if periodic:
        rel = (rel + edge / 2.0) % edge - edge / 2.0
    d = jnp.linalg.norm(rel, axis=-1)
    d = d.at[jnp.diag_indices(d.shape[0])].set(jnp.inf)
    return jnp.argmin(d, axis=-1), jnp.min(d, axis=-1)


def dos(pos, edge, periodic):
    """Mean nearest-neighbour distance, normalised by the max possible separation.

    On a torus per-axis separation caps at edge/2, so D = edge*sqrt(2)/2; the
    paper's 22%/19% are comparable only under that normalisation.

    Not density-normalised: nearest-neighbour distance falls as 1/sqrt(N), so
    DoS compares only across runs with the same population.
    It is also floored at (r_i + r_j)/D, the contact diameter: agents cannot
    overlap, so larger agents cannot reach a lower DoS.
    """
    _, d = nearest_neighbor(pos, edge, periodic)
    D = edge * jnp.sqrt(2.0) / (2.0 if periodic else 1.0)
    return jnp.mean(d / D)


def doa(pos, theta, edge, periodic):
    """Mean ||h_j + h_k|| / 2 over each agent and its nearest conspecific."""
    k, _ = nearest_neighbor(pos, edge, periodic)
    h = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return jnp.mean(jnp.linalg.norm(h + h[k], axis=-1) / 2.0)
