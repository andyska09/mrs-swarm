"""DDPG on a single-agent gymnax env — the learner's regression harness.

Phase 0: proves the update math and the artifact plumbing on a known env. The
predator-prey loop is a separate file; the two share ddpg.py.

The whole run is one jitted lax.scan over episodes, each an inner scan over
steps, with the replay buffer in the carry. Episodes are fixed-length, matching
the paper's regime and ddpg.py's always-bootstrap target.
"""
import flax.struct as struct
import jax
import jax.numpy as jnp

from swarm.algo import ddpg


@struct.dataclass
class Carry:
    agent: ddpg.Agent
    buf: ddpg.Buffer
    norm: ddpg.ObsNorm
    env_state: object
    obs: jnp.ndarray
    key: jnp.ndarray
    eps: jnp.float32
    noise: jnp.float32


def make_train(cfg, env, env_params):
    obs_dim = env.observation_space(env_params).shape[0]
    space = env.action_space(env_params)
    act_dim = space.shape[0]
    algo = ddpg.make_ddpg(cfg, obs_dim, act_dim)

    def to_env(a):      # [-1, 1] -> the env's action range
        return space.low + (a + 1.0) * 0.5 * (space.high - space.low)

    def step(c, _):
        key, k_act, k_env, k_smp = jax.random.split(c.key, 4)
        a = algo.act(c.agent, ddpg.normalize(c.norm, c.obs[None]), k_act, c.eps, c.noise)
        next_obs, env_state, rew, _, _ = env.step_env(k_env, c.env_state, to_env(a[0]), env_params)

        buf = ddpg.insert(c.buf, c.obs[None], a, rew[None], next_obs[None])
        norm = ddpg.update_obsnorm(c.norm, next_obs[None]) if cfg.normalize_obs else c.norm

        def do_update(_):
            o, ac, r, no = ddpg.sample(buf, k_smp, cfg.batch_size)
            return algo.update(c.agent, (ddpg.normalize(norm, o), ac, r, ddpg.normalize(norm, no)))

        def skip(_):
            zero = jnp.float32(0.0)
            return c.agent, {"critic_loss": zero, "actor_loss": zero, "q_mean": zero}

        agent, aux = jax.lax.cond(buf.size >= cfg.learning_starts, do_update, skip, None)
        c = c.replace(agent=agent, buf=buf, norm=norm, env_state=env_state,
                      obs=next_obs, key=key)
        return c, (rew, aux)

    def episode(c, _):
        key, k_reset = jax.random.split(c.key)
        obs, env_state = env.reset_env(k_reset, env_params)
        c = c.replace(key=key, obs=obs, env_state=env_state)
        c, (rews, aux) = jax.lax.scan(step, c, None, length=cfg.episode_len)

        decay = lambda x: jnp.maximum(cfg.expl_min, x - cfg.expl_decay)
        c = c.replace(eps=decay(c.eps), noise=decay(c.noise))
        metrics = {"ep_return": rews.sum(), "eps": c.eps,
                   **{k: v.mean() for k, v in aux.items()}}
        return c, metrics

    def train(key):
        k_init, k_reset, key = jax.random.split(key, 3)
        obs, env_state = env.reset_env(k_reset, env_params)
        c = Carry(agent=algo.init(k_init),
                  buf=ddpg.new_buffer(cfg.buffer_size, obs_dim, act_dim),
                  norm=ddpg.new_obsnorm(obs_dim),
                  env_state=env_state, obs=obs, key=key,
                  eps=jnp.float32(cfg.eps), noise=jnp.float32(cfg.noise))
        c, metrics = jax.lax.scan(episode, c, None, length=cfg.episodes)
        # The buffer stays in the carry: it is large and only useful mid-run.
        return {"agent": c.agent, "obs_norm": c.norm, "metrics": metrics}

    return train
