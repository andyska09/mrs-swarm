"""
PureJaxRL wrappers for gymnax environments.
LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward,
TerminalObsWrapper.
"""
import jax
import jax.numpy as jnp
import flax.struct as struct
from gymnax.environments import environment


class GymnaxWrapper(object):
    """Base wrapper that delegates attribute access to inner env."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


# ─────────────────────────────────────────────────────────────────────────────
# TerminalObsWrapper — captures pre-reset observation for truncation bootstrap
# ─────────────────────────────────────────────────────────────────────────────

class TerminalObsWrapper(GymnaxWrapper):
    """Stash pre-reset observation in info['terminal_obs'] on episode end.

    Required for truncation bootstrapping (SB3-style, Pardo et al. 2018).
    Place between base env and LogWrapper. On non-done steps, terminal_obs
    equals the regular next obs (harmless — masked by truncated_only=0).
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        return self._env.reset(key, params)

    def step(self, key, state, action, params=None):
        obs, new_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        # The env's step_env now emits the true pre-reset observation in
        # info["terminal_obs"] (computed before gymnax's auto-reset select).
        # Only fill in a fallback if a wrapped env doesn't provide it.
        if "terminal_obs" not in info:
            info["terminal_obs"] = obs  # approximate fallback
        return obs, new_state, reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
# LogWrapper — episode return/length tracking
# ─────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.float32
    episode_lengths: jnp.int32
    returned_episode_returns: jnp.float32
    returned_episode_lengths: jnp.int32
    timestep: jnp.int32


class LogWrapper(GymnaxWrapper):
    """Tracks episode returns and lengths via (1-done) arithmetic."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=jnp.int32(0),
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=jnp.int32(0),
            timestep=jnp.int32(0),
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["timestep"] = state.timestep
        return obs, state, reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
# ClipAction
# ─────────────────────────────────────────────────────────────────────────────

class ClipAction(GymnaxWrapper):
    """Clips continuous actions to action space bounds."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        action = jnp.clip(
            action,
            self._env.action_space(params).low,
            self._env.action_space(params).high,
        )
        return self._env.step(key, state, action, params)


# ─────────────────────────────────────────────────────────────────────────────
# VecEnv
# ─────────────────────────────────────────────────────────────────────────────

class VecEnv(GymnaxWrapper):
    """Vectorizes reset and step via vmap."""

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


# ─────────────────────────────────────────────────────────────────────────────
# NormalizeVecObservation — with optional post-normalization clipping
# ─────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class NormalizeVecObsState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.float32
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    """Welford running mean/var normalization on observations.

    Args:
        env: inner environment
        clip_obs: if True, clip normalized obs to [-clip_range, clip_range]
            (CleanRL continuous PPO uses clip_range=10.0)
        clip_range: clipping bound for normalized observations
    """

    def __init__(self, env, clip_obs=False, clip_range=10.0):
        super().__init__(env)
        self.clip_obs = clip_obs
        self.clip_range = clip_range

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = NormalizeVecObsState(
            mean=jnp.zeros_like(obs[0]),
            var=jnp.ones_like(obs[0]),
            count=jnp.float32(1e-4),
            env_state=env_state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]
        state = _update_mean_var_count(state, batch_mean, batch_var, batch_count)
        obs = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
        if self.clip_obs:
            obs = jnp.clip(obs, -self.clip_range, self.clip_range)
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        state = NormalizeVecObsState(
            mean=state.mean,
            var=state.var,
            count=state.count,
            env_state=env_state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]
        state = _update_mean_var_count(state, batch_mean, batch_var, batch_count)
        obs = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
        if self.clip_obs:
            obs = jnp.clip(obs, -self.clip_range, self.clip_range)
        # Also normalize terminal_obs with same stats (for truncation bootstrap)
        if "terminal_obs" in info:
            info["terminal_obs"] = (info["terminal_obs"] - state.mean) / jnp.sqrt(state.var + 1e-8)
            if self.clip_obs:
                info["terminal_obs"] = jnp.clip(info["terminal_obs"], -self.clip_range, self.clip_range)
        return obs, state, reward, done, info


def _update_mean_var_count(state, batch_mean, batch_var, batch_count):
    delta = batch_mean - state.mean
    tot_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / tot_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
    new_var = M2 / tot_count
    return state.replace(mean=new_mean, var=new_var, count=tot_count)


# ─────────────────────────────────────────────────────────────────────────────
# NormalizeVecReward — with optional post-normalization clipping
# ─────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class NormalizeVecRewState:
    mean: jnp.float32
    var: jnp.float32
    count: jnp.float32
    return_val: jnp.ndarray  # (num_envs,)
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    """Normalizes reward by running std of discounted returns.

    Args:
        env: inner environment
        gamma: discount factor for return accumulation
        clip_reward: if True, clip normalized reward to [-clip_range, clip_range]
            (CleanRL continuous PPO uses clip_range=10.0)
        clip_range: clipping bound for normalized rewards
    """

    def __init__(self, env, gamma, clip_reward=False, clip_range=10.0):
        super().__init__(env)
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.clip_range = clip_range

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        batch_size = obs.shape[0]
        state = NormalizeVecRewState(
            mean=jnp.float32(0.0),
            var=jnp.float32(1.0),
            count=jnp.float32(1e-4),
            return_val=jnp.zeros(batch_size),
            env_state=env_state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val)
        batch_var = jnp.var(return_val)
        batch_count = return_val.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count
        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count

        state = NormalizeVecRewState(
            mean=new_mean,
            var=new_var,
            count=tot_count,
            return_val=return_val,
            env_state=env_state,
        )
        reward = reward / jnp.sqrt(state.var + 1e-8)
        if self.clip_reward:
            reward = jnp.clip(reward, -self.clip_range, self.clip_range)
        return obs, state, reward, done, info
