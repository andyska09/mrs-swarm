"""DDPG: actor, critic, replay buffer, one update. Species-agnostic.

Independent DDPG. The critic is Q(o_i, a_i) — local observation and own action,
Li 2023's modification (1). Observations arrive as (batch, obs_dim), so a policy
trained on 10 prey runs unchanged on 50.

The buffer carries no done flag: every environment here has fixed-length
episodes, so the TD target always bootstraps. An env that can terminate early
needs a done mask added here.
"""
from typing import Callable, NamedTuple

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp
import optax


class Actor(nn.Module):
    hidden_dims: tuple
    act_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for d in self.hidden_dims:
            x = nn.relu(nn.Dense(d)(x))
        return nn.tanh(nn.Dense(self.act_dim)(x))   # [-1, 1]; the env scales


class Critic(nn.Module):
    hidden_dims: tuple

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for d in self.hidden_dims:
            x = nn.relu(nn.Dense(d)(x))
        return nn.Dense(1)(x).squeeze(-1)


@struct.dataclass
class Agent:
    actor: dict
    critic: dict
    actor_target: dict
    critic_target: dict
    actor_opt: optax.OptState
    critic_opt: optax.OptState


# ── replay buffer ────────────────────────────────────────────────────────────
# Circular, fixed shapes, lives in a lax.scan carry. Stores RAW observations:
# the normalisation stats drift during training, so scaling happens at sample
# time, under the current stats.

@struct.dataclass
class Buffer:
    obs: jnp.ndarray
    act: jnp.ndarray
    rew: jnp.ndarray
    next_obs: jnp.ndarray
    ptr: jnp.int32
    size: jnp.int32


def new_buffer(capacity, obs_dim, act_dim):
    return Buffer(obs=jnp.zeros((capacity, obs_dim)),
                  act=jnp.zeros((capacity, act_dim)),
                  rew=jnp.zeros((capacity,)),
                  next_obs=jnp.zeros((capacity, obs_dim)),
                  ptr=jnp.int32(0), size=jnp.int32(0))


def insert(buf, obs, act, rew, next_obs):
    """Insert n transitions at once — n is the number of conspecifics this step."""
    cap, n = buf.obs.shape[0], obs.shape[0]
    idx = (buf.ptr + jnp.arange(n)) % cap
    return buf.replace(
        obs=buf.obs.at[idx].set(obs),
        act=buf.act.at[idx].set(act),
        rew=buf.rew.at[idx].set(rew),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        ptr=(buf.ptr + n) % cap,
        size=jnp.minimum(buf.size + n, cap),
    )


def sample(buf, key, batch_size):
    hi = jnp.maximum(buf.size, 1)   # size is 0 until the first insert
    idx = jax.random.randint(key, (batch_size,), 0, hi)
    return buf.obs[idx], buf.act[idx], buf.rew[idx], buf.next_obs[idx]


# ── observation normalisation ────────────────────────────────────────────────

@struct.dataclass
class ObsNorm:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.float32


def new_obsnorm(obs_dim):
    return ObsNorm(jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.float32(1e-4))


def update_obsnorm(n, obs):
    """Welford, batched over the leading axis."""
    bm, bv, bc = obs.mean(0), obs.var(0), obs.shape[0]
    delta = bm - n.mean
    tot = n.count + bc
    m2 = n.var * n.count + bv * bc + delta ** 2 * n.count * bc / tot
    return ObsNorm(n.mean + delta * bc / tot, m2 / tot, tot)


def normalize(n, obs):
    return (obs - n.mean) / jnp.sqrt(n.var + 1e-8)


# ── the algorithm ────────────────────────────────────────────────────────────

class DDPG(NamedTuple):
    init: Callable
    act: Callable
    update: Callable


def make_ddpg(cfg, obs_dim, act_dim):
    actor = Actor(cfg.hidden_dims, act_dim)
    critic = Critic(cfg.hidden_dims)
    actor_tx = optax.adam(cfg.lr_actor)
    critic_tx = optax.adam(cfg.lr_critic)

    def init(key):
        ka, kc = jax.random.split(key)
        o, a = jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim))
        ap, cp = actor.init(ka, o), critic.init(kc, o, a)
        return Agent(ap, cp, ap, cp, actor_tx.init(ap), critic_tx.init(cp))

    def act(agent, obs, key, eps, noise):
        """obs (n, obs_dim), already normalised -> action (n, act_dim) in [-1, 1]."""
        k_noise, k_pick, k_rand = jax.random.split(key, 3)
        a = actor.apply(agent.actor, obs)
        a = a + noise * jax.random.normal(k_noise, a.shape)
        rand = jax.random.uniform(k_rand, a.shape, minval=-1.0, maxval=1.0)
        # eps swaps the entire action vector for a uniform sample.
        take_rand = jax.random.bernoulli(k_pick, eps, a.shape[:-1])[..., None]
        return jnp.clip(jnp.where(take_rand, rand, a), -1.0, 1.0)

    def update(agent, batch):
        obs, act_b, rew, next_obs = batch
        a_next = actor.apply(agent.actor_target, next_obs)
        y = rew + cfg.gamma * critic.apply(agent.critic_target, next_obs, a_next)

        def critic_loss(p):
            q = critic.apply(p, obs, act_b)
            return ((q - y) ** 2).mean(), q.mean()

        (c_loss, q_mean), g = jax.value_and_grad(critic_loss, has_aux=True)(agent.critic)
        upd, critic_opt = critic_tx.update(g, agent.critic_opt)
        new_critic = optax.apply_updates(agent.critic, upd)

        def actor_loss(p):
            return -critic.apply(new_critic, obs, actor.apply(p, obs)).mean()

        a_loss, g = jax.value_and_grad(actor_loss)(agent.actor)
        upd, actor_opt = actor_tx.update(g, agent.actor_opt)
        new_actor = optax.apply_updates(agent.actor, upd)

        def soft(target, source):
            return jax.tree.map(lambda t, s: (1 - cfg.tau) * t + cfg.tau * s, target, source)

        agent = Agent(new_actor, new_critic,
                      soft(agent.actor_target, new_actor),
                      soft(agent.critic_target, new_critic),
                      actor_opt, critic_opt)
        return agent, {"critic_loss": c_loss, "actor_loss": a_loss, "q_mean": q_mean}

    return DDPG(init, act, update)
