"""Predator-prey environment. Spec section 1.

Agents are an array predators first: `pos[:n_pred]`, then `pos[n_pred:]`.

`cfg` is an EnvConfig and is passed to jit as a static argument.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

SPAWN_TRIES = 100  # bound on the rejection sampler; dense arenas give up and overlap
EPS = 1e-12  # divide-by-zero guard


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
    return _per_agent(cfg, cfg.max_speed_pred, cfg.max_speed_prey)


def max_accs(cfg):
    return _per_agent(cfg, cfg.max_acc * cfg.pred_acc_scale, cfg.max_acc)


def max_ang_vels(cfg):
    return _per_agent(cfg, cfg.max_ang_vel * cfg.pred_turn_scale, cfg.max_ang_vel)


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


def _lattice_pos(key, cfg):
    """Prey on a random square grid, predators uniform. Spawn for the formation test."""
    n1, s, half = cfg.n_prey, cfg.spawn_spacing, cfg.edge / 2
    side = math.ceil(math.sqrt(n1))
    k_c, k_j, k_p = jax.random.split(key, 3)
    ij = jnp.stack(jnp.meshgrid(jnp.arange(side), jnp.arange(side)), -1).reshape(-1, 2)
    grid = (ij[:n1] - (side - 1) / 2.0) * s
    prey = (
        jax.random.uniform(k_c, (2,), minval=-half, maxval=half)
        + grid
        + jax.random.uniform(k_j, (n1, 2), minval=-0.1 * s, maxval=0.1 * s)
    )
    pred = jax.random.uniform(k_p, (cfg.n_pred, 2), minval=-half, maxval=half)
    return _wrap_pos(jnp.concatenate([pred, prey]), cfg)


def _spawn_pos(key, cfg):
    if cfg.spawn == "lattice":
        return _lattice_pos(key, cfg)
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


def wrap_angle(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _wrap_pos(pos, cfg):
    if cfg.boundary == "torus":
        return (pos + cfg.edge / 2) % cfg.edge - cfg.edge / 2
    return pos


def scale_action(action, cfg):
    """Actor output in [-1, 1]^2 -> physical (a_F, a_R). Spec 1.3."""
    a = jnp.clip(action, -1.0, 1.0)
    return (a[:, 0] + 1.0) * 0.5 * max_accs(cfg), a[:, 1] * max_ang_vels(cfg)


def contact_force(pos, cfg):
    """Hooke on overlap, summed over contacts: f_a = sum_j f_a,j."""
    r = radii(cfg)
    rel = delta(pos[:, None, :], pos[None, :, :], cfg)  # rel[i, j] = pos_j - pos_i
    dist = jnp.linalg.norm(rel, axis=-1)
    overlap = jnp.maximum(r[:, None] + r[None, :] - dist, 0.0)
    overlap = overlap * (1.0 - jnp.eye(pos.shape[0]))
    away = -rel / (dist[..., None] + EPS)
    return cfg.stiffness * jnp.sum(overlap[..., None] * away, axis=1)


def wall_force(pos, cfg):
    """-> (force, per-agent 'touching a wall'). Both zero on a torus."""
    if cfg.boundary == "torus":
        return jnp.zeros_like(pos), jnp.zeros(pos.shape[0], dtype=bool)
    r, half = radii(cfg)[:, None], cfg.edge / 2
    lo = jnp.maximum((-half + r) - pos, 0.0)
    hi = jnp.maximum(pos - (half - r), 0.0)
    return cfg.stiffness * (lo - hi), jnp.any((lo > 0.0) | (hi > 0.0), axis=-1)


def step(state, action, cfg):
    a_f, a_r = scale_action(action, cfg)

    theta = wrap_angle(state.theta + a_r * cfg.dt)
    h = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    force = (
        a_f[:, None] * h
        - cfg.drag * state.vel
        + contact_force(state.pos, cfg)
        + wall_force(state.pos, cfg)[0]
    )
    vel = state.vel + force * cfg.dt / masses(cfg)[:, None]
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    vel = vel * jnp.minimum(1.0, max_speeds(cfg)[:, None] / (speed + EPS))

    # The new velocity - see choices.md, differs from the paper eq.
    pos = _wrap_pos(state.pos + vel * cfg.dt, cfg)
    return State(pos=pos, vel=vel, theta=theta, time=state.time + 1)


def act_dim(cfg):
    """d_a = 2: (a_F, a_R)"""
    return 2


def heading_dim(cfg):
    return 2 if cfg.heading_encoding == "unit" else 1


def obs_dim(cfg):
    """54 with unit headings, 41 with raw angles"""
    return 4 + heading_dim(cfg) + 2 * cfg.n_neighbors * (2 + heading_dim(cfg))


def heading(theta, cfg):
    if cfg.heading_encoding == "unit":
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return theta[..., None]


def _nearest(rel, head, dist, k, radius):
    """One species' neighbours -> k fixed slots, nearest first, zeroed beyond `radius`.

    return: (features (n, k, 2 + heading_dim), mask (n, k)). If the species has
    fewer than k members the spare slots are padded at infinite distance.
    """
    n, m = dist.shape
    head = jnp.broadcast_to(head[None], (n, m, head.shape[-1]))
    pad = max(0, k - m)
    if pad:
        dist = jnp.concatenate([dist, jnp.full((n, pad), jnp.inf)], axis=1)
        rel = jnp.concatenate([rel, jnp.zeros((n, pad, 2))], axis=1)
        head = jnp.concatenate([head, jnp.zeros((n, pad, head.shape[-1]))], axis=1)

    order = jnp.argsort(dist, axis=1)[:, :k]
    mask = jnp.take_along_axis(dist, order, axis=1) <= radius
    feat = jnp.concatenate(
        [
            jnp.take_along_axis(rel, order[..., None], axis=1),
            jnp.take_along_axis(head, order[..., None], axis=1),
        ],
        axis=-1,
    )
    return feat * mask[..., None], mask


def observe(state, cfg):
    """return: (N, obs_dim). Own state, then <= k predators, then <= k prey."""
    n, n0, k = n_agents(cfg), cfg.n_pred, cfg.n_neighbors
    head = heading(state.theta, cfg)
    rel = delta(state.pos[:, None, :], state.pos[None, :, :], cfg)
    dist = jnp.linalg.norm(rel, axis=-1)
    dist = dist.at[jnp.diag_indices(n)].set(jnp.inf)  # never observe yourself

    pred, _ = _nearest(rel[:, :n0], head[:n0], dist[:, :n0], k, perception(cfg))
    prey, _ = _nearest(rel[:, n0:], head[n0:], dist[:, n0:], k, perception(cfg))
    own = jnp.concatenate([state.pos, state.vel, head], axis=-1)
    return jnp.concatenate([own, pred.reshape(n, -1), prey.reshape(n, -1)], axis=-1)


def contacts(pos, cfg):
    """(N, N) bool: discs that overlap. Capture IS contact,"""
    r = radii(cfg)
    dist = jnp.linalg.norm(delta(pos[:, None, :], pos[None, :, :], cfg), axis=-1)
    return (dist < r[:, None] + r[None, :]) & ~jnp.eye(pos.shape[0], dtype=bool)


def reward(state, action, cfg):
    """-> (reward (N,), info) An agent touching several adversaries still scores once."""
    n0 = cfg.n_pred
    cross = contacts(state.pos, cfg)[:n0, n0:]  # (n_pred, n_prey)
    hunting = jnp.any(cross, axis=1).astype(jnp.float32)
    hunted = jnp.any(cross, axis=0).astype(jnp.float32)
    survival = cfg.catch_reward * jnp.concatenate([hunting, -hunted])

    a_f, a_r = scale_action(action, cfg)
    scale = _per_agent(cfg, 1.0, cfg.prey_cost_scale)
    movement = -scale * (cfg.cost_af * jnp.abs(a_f) + cfg.cost_ar * jnp.abs(a_r))
    wall = -cfg.boundary_penalty * wall_force(state.pos, cfg)[1].astype(jnp.float32)

    info = {
        "survival": survival,
        "movement": movement,
        "wall": wall,
        "captures": hunted.sum(),
    }
    return survival + movement + wall, info


def rollout(key, cfg, policy):
    """One episode under `policy(key, obs) -> (N, 2)` in [-1, 1].

    return: (states, rewards, info), time on the leading axis.
    """

    def tick(carry, _):
        state, key = carry
        key, k = jax.random.split(key)
        action = policy(k, observe(state, cfg))
        state = step(state, action, cfg)
        r, info = reward(state, action, cfg)
        return (state, key), (state, r, info)

    _, out = jax.lax.scan(tick, (reset(key, cfg), key), None, length=cfg.episode_len)
    return out
