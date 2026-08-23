"""Predator-prey swarm environment (Li 2023, sections 2-3 and appendix B).

Every quantity is per-agent. Agents are packed into one array, predators first:
`pos[:n_pred]` is the predator block, `pos[n_pred:]` the prey block. Population
sizes are static, so evaluating on 50 prey recompiles.

Actions arrive in [-1, 1]^2 and are scaled here: a_F is forward-only,
[-1,1] -> [0, max_acc]; a_R -> [-max_ang_vel, max_ang_vel].

Prey survive capture and stay in the population. Contact bleeds continuously:
-1 for every step a prey touches a predator, back to zero on separation.
Episodes are a fixed 100 steps.

Unstated in the paper, chosen here:
  - agent radii; contact is the capture condition, so they set the reward scale
  - initial velocity zero, positions uniform, headings uniform
  - a predator touching several prey scores +1 for the step
"""
import flax.struct as struct
import jax
import jax.numpy as jnp

from swarm.envs.dynamics import step_motion


@struct.dataclass
class EnvState:
    pos: jnp.ndarray      # (N, 2) predators first
    vel: jnp.ndarray      # (N, 2)
    theta: jnp.ndarray    # (N,)
    time: jnp.int32


@struct.dataclass
class EnvParams:
    # Static: resolved at trace time, so each combination compiles its own step.
    n_pred: int = struct.field(pytree_node=False, default=3)
    n_prey: int = struct.field(pytree_node=False, default=10)
    boundary: str = struct.field(pytree_node=False, default="torus")     # "torus" | "walls"
    motion_mode: str = struct.field(pytree_node=False, default="unicycle2d")
    n_neighbors: int = struct.field(pytree_node=False, default=6)        # Ballerini 2008 topological limit
    episode_len: int = struct.field(pytree_node=False, default=100)

    # Table 1.
    edge: float = 2.0
    dt: float = 0.1
    mass: float = 1.0
    drag: float = 2.0
    stiffness: float = 50.0
    max_acc: float = 1.0
    max_ang_vel: float = 0.5
    speed_pred: float = 0.5
    speed_prey: float = 0.5
    perception: float = 2.0        # R = D = env edge in the main runs

    # Not in the paper. 0.06 + 0.04 = 5% of the box; at max speed an agent covers
    # 0.05 m per step, half the contact diameter, so contact is always detected.
    radius_pred: float = 0.06
    radius_prey: float = 0.04

    # Section 3.4.
    catch_reward: float = 1.0
    cost_af: float = 0.01
    cost_ar: float = 0.1
    boundary_penalty: float = 0.1  # walls only


def n_agents(params):
    return params.n_pred + params.n_prey


def obs_dim(params):
    return 6 + 2 * params.n_neighbors * 4


def radii(params):
    return jnp.concatenate([jnp.full(params.n_pred, params.radius_pred),
                            jnp.full(params.n_prey, params.radius_prey)])


def max_speeds(params):
    return jnp.concatenate([jnp.full(params.n_pred, params.speed_pred),
                            jnp.full(params.n_prey, params.speed_prey)])


# ── geometry ─────────────────────────────────────────────────────────────────

def pairwise(pos, params):
    """rel[i, j] = x_j - x_i under the minimum-image convention on a torus."""
    rel = pos[None, :, :] - pos[:, None, :]
    if params.boundary == "torus":
        rel = (rel + params.edge / 2.0) % params.edge - params.edge / 2.0
    return rel, jnp.linalg.norm(rel, axis=-1)


def _wrap(pos, params):
    if params.boundary == "torus":
        return (pos + params.edge / 2.0) % params.edge - params.edge / 2.0
    return pos


def _contact_force(rel, dist, r, params):
    """Hooke on overlap, summed over contacts (eq: f_a = sum_j f_a,j)."""
    overlap = jnp.maximum(r[:, None] + r[None, :] - dist, 0.0)
    overlap = overlap * (1.0 - jnp.eye(dist.shape[0]))
    n_hat = -rel / (dist[..., None] + 1e-8)          # pushes i away from j
    return jnp.sum(params.stiffness * overlap[..., None] * n_hat, axis=1)


def _boundary_force(pos, r, params):
    """-> (force, per-agent boolean 'touching a wall'). Zero on a torus."""
    if params.boundary == "torus":
        return jnp.zeros_like(pos), jnp.zeros(pos.shape[0], dtype=bool)
    half = params.edge / 2.0
    lo = jnp.maximum((-half + r[:, None]) - pos, 0.0)
    hi = jnp.maximum(pos - (half - r[:, None]), 0.0)
    touching = jnp.any((lo > 0.0) | (hi > 0.0), axis=-1)
    return params.stiffness * (lo - hi), touching


# ── observation ──────────────────────────────────────────────────────────────

def _gather(rel, h_other, dist, k, R):
    """Nearest k of a block, nearest-first, zero-padded. -> (feat (n,k,4), mask (n,k))."""
    n, m = dist.shape
    h_other = jnp.broadcast_to(h_other[None, :, :], (n, m, 2))
    pad = max(0, k - m)
    if pad:
        dist = jnp.concatenate([dist, jnp.full((n, pad), jnp.inf)], axis=-1)
        rel = jnp.concatenate([rel, jnp.zeros((n, pad, 2))], axis=1)
        h_other = jnp.concatenate([h_other, jnp.zeros((n, pad, 2))], axis=1)

    order = jnp.argsort(dist, axis=-1)[:, :k]
    mask = jnp.take_along_axis(dist, order, axis=-1) <= R
    feat = jnp.concatenate([jnp.take_along_axis(rel, order[..., None], axis=1),
                            jnp.take_along_axis(h_other, order[..., None], axis=1)], axis=-1)
    return feat * mask[..., None], mask


def get_obs(state, params):
    """{'self': (N,6), 'neighbors': (N,2K,4), 'mask': (N,2K)}, ALLIES FIRST.

    Ally-first ordering makes the two species' observations semantically
    identical, so one env emits both. Headings are unit vectors: theta wraps at
    +-pi, and an MLP fed that discontinuity fails to learn turning.
    """
    n0, K, R = params.n_pred, params.n_neighbors, params.perception
    h = jnp.stack([jnp.cos(state.theta), jnp.sin(state.theta)], axis=-1)
    rel, dist = pairwise(state.pos, params)
    dist = dist.at[jnp.diag_indices(dist.shape[0])].set(jnp.inf)   # never observe yourself

    p_ally, m_pa = _gather(rel[:n0, :n0], h[:n0], dist[:n0, :n0], K, R)
    p_adv, m_pv = _gather(rel[:n0, n0:], h[n0:], dist[:n0, n0:], K, R)
    y_ally, m_ya = _gather(rel[n0:, n0:], h[n0:], dist[n0:, n0:], K, R)
    y_adv, m_yv = _gather(rel[n0:, :n0], h[:n0], dist[n0:, :n0], K, R)

    return {
        "self": jnp.concatenate([state.pos, state.vel, h], axis=-1),
        "neighbors": jnp.concatenate([jnp.concatenate([p_ally, p_adv], axis=1),
                                      jnp.concatenate([y_ally, y_adv], axis=1)], axis=0),
        "mask": jnp.concatenate([jnp.concatenate([m_pa, m_pv], axis=1),
                                 jnp.concatenate([m_ya, m_yv], axis=1)], axis=0),
    }


def flatten_obs(obs):
    """The paper's MLP input: self features then the flattened neighbour block."""
    n = obs["self"].shape[0]
    return jnp.concatenate([obs["self"], obs["neighbors"].reshape(n, -1)], axis=-1)


# ── reward ───────────────────────────────────────────────────────────────────

def compute_reward(state, action_phys, params):
    """action_phys is [a_F, a_R] in physical units. -> (reward (N,), info)."""
    n0 = params.n_pred
    _, dist = pairwise(state.pos, params)
    r = radii(params)
    contact = dist < (r[:, None] + r[None, :])
    cross = contact[:n0, n0:]                       # (n_pred, n_prey)

    pred_r = params.catch_reward * jnp.any(cross, axis=1)
    prey_r = -params.catch_reward * jnp.any(cross, axis=0)
    survival = jnp.concatenate([pred_r, prey_r]).astype(jnp.float32)

    movement = -(params.cost_af * jnp.abs(action_phys[:, 0])
                 + params.cost_ar * jnp.abs(action_phys[:, 1]))

    _, touching = _boundary_force(state.pos, r, params)
    wall = -params.boundary_penalty * touching.astype(jnp.float32)

    # The two reward terms are logged apart: for an untrained agent the movement
    # cost dominates survival, and only the split shows that.
    info = {"captures": jnp.sum(cross.any(axis=0)).astype(jnp.float32),
            "wall_contacts": jnp.sum(touching).astype(jnp.float32),
            "survival": survival, "movement": movement + wall}
    return survival + movement + wall, info


# ── env API ──────────────────────────────────────────────────────────────────

def reset(key, params):
    k_pos, k_th = jax.random.split(key)
    n, half = n_agents(params), params.edge / 2.0
    state = EnvState(pos=jax.random.uniform(k_pos, (n, 2), minval=-half, maxval=half),
                     vel=jnp.zeros((n, 2)),
                     theta=jax.random.uniform(k_th, (n,), minval=-jnp.pi, maxval=jnp.pi),
                     time=jnp.int32(0))
    return get_obs(state, params), state


def step(state, action, params):
    """action (N, 2) in [-1, 1]. Deterministic, so it takes no key."""
    action = jnp.clip(action, -1.0, 1.0)
    a_f = (action[:, 0] + 1.0) * 0.5 * params.max_acc      # forward only
    a_r = action[:, 1] * params.max_ang_vel
    action_phys = jnp.stack([a_f, a_r], axis=-1)

    r = radii(params)
    rel, dist = pairwise(state.pos, params)
    force = _contact_force(rel, dist, r, params) + _boundary_force(state.pos, r, params)[0]

    pos, vel, theta = step_motion(state.pos, state.vel, state.theta,
                                  action_phys, force, params, max_speeds(params))
    new_state = EnvState(pos=_wrap(pos, params), vel=vel, theta=theta, time=state.time + 1)

    reward, info = compute_reward(new_state, action_phys, params)
    done = new_state.time >= params.episode_len
    return get_obs(new_state, params), new_state, reward, done, info


def rollout(key, params, policy):
    """Run one episode under `policy(key, obs, state, params) -> action (N,2)`.

    Returns the stacked state trajectory (leading axis = time) and per-step
    rewards. Used by render.py and the phase-1 gates; the learner has its own
    loop, which also carries a replay buffer.
    """
    obs, state = reset(key, params)

    def one(carry, _):
        state, obs, key = carry
        key, k = jax.random.split(key)
        obs, state, reward, _, info = step(state, policy(k, obs, state, params), params)
        return (state, obs, key), (state, reward, info)

    _, (traj, rewards, info) = jax.lax.scan(one, (state, obs, key), None,
                                            length=params.episode_len)
    return traj, rewards, info


# Presets — the only place values are chosen.
PRESETS = {
    "torus": EnvParams(),                                    # the main flocking result
    "eval50": EnvParams(n_prey=50),                          # the paper's evaluation population
    "npred0": EnvParams(n_pred=0),                           # the control that must stay flat
    "npred1": EnvParams(n_pred=1),
    "speed_53": EnvParams(speed_prey=0.3),                   # predator : prey = 5 : 3
    "speed_35": EnvParams(speed_pred=0.3),                   # 3 : 5
    "perception_23": EnvParams(perception=4.0 / 3.0),
    "perception_13": EnvParams(perception=2.0 / 3.0),
    "walls": EnvParams(boundary="walls"),                    # phase 4, swirling
}


def get_env_params(preset: str = "torus") -> EnvParams:
    assert preset in PRESETS, f"Unknown preset {preset!r}. Available: {list(PRESETS)}"
    return PRESETS[preset]
