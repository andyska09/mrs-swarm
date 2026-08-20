"""
Interceptor2D — a 2D pursuit environment (gymnax API).

Pursuer: point mass, acceleration-controlled, speed-capped, with drag.
Target:  chosen by the STATIC field `evader_mode` (resolved at trace time —
         each mode compiles its own step, there is no branch at runtime):
             0 = straight (constant velocity)      <- default
             1 = weave (evades, sinusoidal sideways component)

Obs (7): rel_pos(2), rel_vel(2), own_vel(2), closing_speed(1). No time in obs.
Action (2): in [-1, 1], scaled to +-max_acc.

Termination — this is the part people get wrong:
    capture (dist < capture_radius)  -> TRUE terminal
    miss    (dist > miss_radius)     -> TRUE terminal
    timeout (t >= max_steps)         -> TRUNCATION, not a terminal
gymnax auto-resets on done = terminated | truncated, which throws away the real
last observation. We emit it in info["terminal_obs"] so the trainer can
bootstrap V(s_T) at truncations.

Reward = progress + fuel + capture bonus + miss penalty:
    dist_scale * (prev_dist - curr_dist)   dense: reward for closing distance
  - action_penalty * ||a||^2               small fuel cost
  + capture_bonus  if captured
  + miss_penalty   if missed
"""
import jax
import jax.numpy as jnp
import flax.struct as struct
from gymnax.environments import environment, spaces


@struct.dataclass
class RewardConfig:
    dist_scale: float = 1.0
    action_penalty: float = 0.01
    capture_bonus: float = 10.0
    miss_penalty: float = -10.0
    capture_radius: float = 0.5
    miss_radius: float = 20.0


def compute_reward(state, next_state, action, cfg: RewardConfig):
    rel = next_state.pos_t - next_state.pos_p
    curr_dist = jnp.sqrt(jnp.sum(rel ** 2) + 1e-8)
    captured = curr_dist < cfg.capture_radius
    missed = curr_dist > cfg.miss_radius

    r_progress = (state.prev_dist - curr_dist) * cfg.dist_scale
    r_action = -cfg.action_penalty * jnp.sum(action ** 2)
    r_capture = jnp.where(captured, cfg.capture_bonus, 0.0)
    r_miss = jnp.where(missed, cfg.miss_penalty, 0.0)

    reward = r_progress + r_action + r_capture + r_miss
    info = {"r_progress": r_progress, "r_action": r_action,
            "r_capture": r_capture, "r_miss": r_miss}
    return reward, info


@struct.dataclass
class EnvState(environment.EnvState):
    pos_p: jnp.ndarray       # (2,) pursuer position
    vel_p: jnp.ndarray       # (2,) pursuer velocity
    pos_t: jnp.ndarray       # (2,) target position
    vel_t: jnp.ndarray       # (2,) target velocity
    prev_dist: jnp.float32   # for the progress reward
    evader_phase: jnp.float32
    time: jnp.int32


@struct.dataclass
class EnvParams(environment.EnvParams):
    # Pursuer physics
    dt: float = 0.05
    max_acc: float = 5.0
    pursuer_max_speed: float = 8.0
    damping: float = 0.1
    # Target
    evader_mode: int = struct.field(pytree_node=False, default=0)
    target_evade_acc: float = 4.0     # weave only
    target_max_speed: float = 5.0     # weave only
    # Episode
    max_steps_in_episode: int = 500
    # Spawn ranges
    target_dist_min: float = 5.0
    target_dist_max: float = 15.0
    target_speed_min: float = 0.5
    target_speed_max: float = 2.0
    pursuer_vel_range: float = 0.5
    # Reward
    reward_config: RewardConfig = RewardConfig()


class Interceptor2D(environment.Environment):
    obs_size = 7
    num_actions = 2

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _target_velocity(self, state, params):
        if params.evader_mode == 0:
            return state.vel_t
        # weave: run away with a sinusoidal sideways component
        rel = state.pos_t - state.pos_p
        away = rel / jnp.sqrt(jnp.sum(rel ** 2) + 1e-8)
        perp = jnp.array([-away[1], away[0]])
        phase = jnp.sin(state.time.astype(jnp.float32) * 0.3 + state.evader_phase)
        d = away * 0.6 + perp * phase * 0.4
        d = d / jnp.sqrt(jnp.sum(d ** 2) + 1e-8)
        v = state.vel_t + d * params.target_evade_acc * params.dt
        s = jnp.sqrt(jnp.sum(v ** 2) + 1e-8)
        return jnp.where(s > params.target_max_speed, v * params.target_max_speed / s, v)

    def step_env(self, key, state, action, params):
        action = jnp.clip(action, -1.0, 1.0)
        acc = action * params.max_acc

        # semi-implicit Euler + drag + speed cap
        v = state.vel_p + (acc - params.damping * state.vel_p) * params.dt
        s = jnp.sqrt(jnp.sum(v ** 2) + 1e-8)
        v = jnp.where(s > params.pursuer_max_speed, v * params.pursuer_max_speed / s, v)
        pos_p = state.pos_p + v * params.dt

        vel_t = self._target_velocity(state, params)
        pos_t = state.pos_t + vel_t * params.dt

        new_state = EnvState(
            pos_p=pos_p, vel_p=v, pos_t=pos_t, vel_t=vel_t,
            prev_dist=jnp.sqrt(jnp.sum((state.pos_t - state.pos_p) ** 2) + 1e-8),
            evader_phase=state.evader_phase,
            time=state.time + jnp.int32(1),
        )

        reward, reward_info = compute_reward(state, new_state, action, params.reward_config)

        terminated = self.is_terminal(new_state, params)
        truncated = new_state.time >= params.max_steps_in_episode
        done = terminated | truncated

        info = {
            "terminated": terminated,
            "truncated": truncated,
            "terminal_obs": self.get_obs(new_state, params),   # pre-reset obs
            **reward_info,
        }
        return (jax.lax.stop_gradient(self.get_obs(new_state, params)),
                jax.lax.stop_gradient(new_state), reward, done, info)

    def reset_env(self, key, params):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        pos_p = jnp.zeros(2)
        vel_p = jax.random.uniform(k1, (2,), minval=-params.pursuer_vel_range,
                                   maxval=params.pursuer_vel_range)
        angle = jax.random.uniform(k2, (), minval=0.0, maxval=2.0 * jnp.pi)
        dist = jax.random.uniform(k3, (), minval=params.target_dist_min,
                                  maxval=params.target_dist_max)
        pos_t = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * dist
        t_angle = jax.random.uniform(k4, (), minval=0.0, maxval=2.0 * jnp.pi)
        t_speed = jax.random.uniform(k5, (), minval=params.target_speed_min,
                                     maxval=params.target_speed_max)
        vel_t = jnp.array([jnp.cos(t_angle), jnp.sin(t_angle)]) * t_speed
        evader_phase = jax.random.uniform(k6, (), minval=0.0, maxval=2.0 * jnp.pi)
        state = EnvState(
            pos_p=pos_p, vel_p=vel_p, pos_t=pos_t, vel_t=vel_t,
            prev_dist=jnp.sqrt(jnp.sum((pos_t - pos_p) ** 2) + 1e-8),
            evader_phase=evader_phase, time=jnp.int32(0),
        )
        return self.get_obs(state, params), state

    def get_obs(self, state, params=None, key=None):
        rel_pos = state.pos_t - state.pos_p
        rel_vel = state.vel_t - state.vel_p
        dist = jnp.sqrt(jnp.sum(rel_pos ** 2) + 1e-8)
        closing = -jnp.sum(rel_pos * rel_vel) / dist
        return jnp.concatenate([rel_pos, rel_vel, state.vel_p, jnp.array([closing])])

    def is_terminal(self, state, params):
        rel = state.pos_t - state.pos_p
        dist = jnp.sqrt(jnp.sum(rel ** 2) + 1e-8)
        cfg = params.reward_config
        return (dist < cfg.capture_radius) | (dist > cfg.miss_radius)

    @property
    def name(self):
        return "Interceptor2D-v0"

    def action_space(self, params=None):
        return spaces.Box(-jnp.ones(2), jnp.ones(2), (2,), jnp.float32)

    def observation_space(self, params=None):
        high = jnp.full(self.obs_size, jnp.inf)
        return spaces.Box(-high, high, (self.obs_size,), jnp.float32)


# Presets — the only place values are chosen.
PRESETS = {
    "straight": EnvParams(evader_mode=0),   # constant-velocity target. Start here.
    "weave": EnvParams(evader_mode=1),      # evading target. Harder.
}


def get_env_params(preset: str = "straight") -> EnvParams:
    assert preset in PRESETS, f"Unknown preset {preset!r}. Available: {list(PRESETS)}"
    return PRESETS[preset]
