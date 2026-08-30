"""MADDPG coevolution. Spec 3 and 4.

The three deviations from standard MADDPG:

- a decentralised critic Q(o_i, a_i), actor and critic shared within a species but not across

- one replay buffer per species. Both species learn concurrently.

One environment, one gradient step per species per environment step

Everything lives in a lax.scan carry, so a run is one XLA program.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from swarm_simple.algo import buffer, networks
from swarm_simple.envs import metrics, scripted
from swarm_simple.envs import predator_prey as pp

OPTIMIZER = {"adam": optax.adam, "sgd": optax.sgd}
NO_UPDATE = {k: jnp.float32(0.0) for k in ("critic_loss", "actor_loss", "q")}


class Species(NamedTuple):
    actor: dict
    critic: dict
    actor_target: dict
    critic_target: dict
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    buf: buffer.Buffer


class Carry(NamedTuple):
    pred: Species
    prey: Species
    state: pp.State
    obs: jnp.ndarray
    key: jnp.ndarray
    eps: jnp.ndarray
    noise: jnp.ndarray
    step: jnp.ndarray


def make_train(exp):
    env_cfg, tr, n0 = exp.env, exp.train, exp.env.n_pred
    d_o, d_a = pp.obs_dim(env_cfg), pp.act_dim(env_cfg)
    actor, critic = networks.build(exp.model, d_a)
    actor_tx = OPTIMIZER[tr.optimizer](tr.lr_actor)
    critic_tx = OPTIMIZER[tr.optimizer](tr.lr_critic)

    def init(key):
        k_a, k_c = jax.random.split(key)
        obs, action = jnp.zeros((1, d_o)), jnp.zeros((1, d_a))
        a_params, c_params = actor.init(k_a, obs), critic.init(k_c, obs, action)
        return Species(
            actor=a_params,
            critic=c_params,
            actor_target=a_params,
            critic_target=c_params,
            actor_opt=actor_tx.init(a_params),
            critic_opt=critic_tx.init(c_params),
            buf=buffer.empty(tr.buffer_size, d_o, d_a),
        )

    def act(sp, obs, key, eps, noise):
        """-> (action, logits) mu(o) + N, or a uniform action with probability eps"""
        k_noise, k_pick, k_uniform = jax.random.split(key, 3)
        n = obs.shape[0]
        mu, z = actor.apply(sp.actor, obs)
        a = mu + noise * jax.random.normal(k_noise, (n, d_a))
        uniform = jax.random.uniform(k_uniform, (n, d_a), minval=-1.0, maxval=1.0)
        explore = jax.random.bernoulli(k_pick, eps, (n, 1))
        return jnp.clip(jnp.where(explore, uniform, a), -1.0, 1.0), z

    def learn(sp, key):
        obs, action, reward, next_obs = buffer.sample(sp.buf, key, tr.batch_size)
        next_action, _ = actor.apply(sp.actor_target, next_obs)
        y = reward + tr.gamma * critic.apply(sp.critic_target, next_obs, next_action)

        def critic_loss(params):
            q = critic.apply(params, obs, action)
            return ((q - y) ** 2).mean(), q.mean()

        (c_loss, q), grad = jax.value_and_grad(critic_loss, has_aux=True)(sp.critic)
        upd, critic_opt = critic_tx.update(grad, sp.critic_opt)
        new_critic = optax.apply_updates(sp.critic, upd)

        def actor_loss(params):
            a, z = actor.apply(params, obs)
            reg = tr.actor_reg * (z**2).mean()
            return -critic.apply(new_critic, obs, a).mean() + reg

        a_loss, grad = jax.value_and_grad(actor_loss)(sp.actor)
        upd, actor_opt = actor_tx.update(grad, sp.actor_opt)
        new_actor = optax.apply_updates(sp.actor, upd)

        def track(target, online):
            return jax.tree.map(
                lambda t, o: (1 - tr.tau) * t + tr.tau * o, target, online
            )

        sp = sp._replace(
            actor=new_actor,
            critic=new_critic,
            actor_target=track(sp.actor_target, new_actor),
            critic_target=track(sp.critic_target, new_critic),
            actor_opt=actor_opt,
            critic_opt=critic_opt,
        )
        return sp, {"critic_loss": c_loss, "actor_loss": a_loss, "q": q}

    def species_metrics(pos, theta):
        if pos.shape[0] < 2:  # a lone agent has no nearest conspecific
            return jnp.float32(jnp.nan), jnp.float32(jnp.nan)
        return metrics.dos(pos, env_cfg), metrics.doa(pos, theta, env_cfg)

    def env_step(carry, _):
        key, k_pred, k_prey, k_lp, k_ly = jax.random.split(carry.key, 5)
        if tr.scripted_predator:
            a_pred, z_pred = scripted.predator(carry.state, env_cfg), jnp.zeros(
                (n0, d_a)
            )
        else:
            a_pred, z_pred = act(
                carry.pred, carry.obs[:n0], k_pred, carry.eps, carry.noise
            )
        a_prey, z_prey = act(carry.prey, carry.obs[n0:], k_prey, carry.eps, carry.noise)
        action = jnp.concatenate([a_pred, a_prey])

        state = pp.step(carry.state, action, env_cfg)
        reward, info = pp.reward(state, action, env_cfg)
        next_obs = pp.observe(state, env_cfg)

        pred = carry.pred._replace(
            buf=buffer.insert(
                carry.pred.buf, carry.obs[:n0], a_pred, reward[:n0], next_obs[:n0]
            )
        )
        prey = carry.prey._replace(
            buf=buffer.insert(
                carry.prey.buf, carry.obs[n0:], a_prey, reward[n0:], next_obs[n0:]
            )
        )

        warm = carry.step >= tr.learning_starts
        pred, p_aux = jax.lax.cond(
            warm & (n0 > 0) & (not tr.scripted_predator),
            lambda: learn(pred, k_lp),
            lambda: (pred, NO_UPDATE),
        )
        prey, y_aux = jax.lax.cond(
            warm, lambda: learn(prey, k_ly), lambda: (prey, NO_UPDATE)
        )

        dos, doa = species_metrics(state.pos[n0:], state.theta[n0:])
        pred_dos, pred_doa = species_metrics(state.pos[:n0], state.theta[:n0])
        nan = jnp.float32(jnp.nan)
        # Physical units: a_F in [0, max_acc], |a_R| in [0, max_ang_vel].
        a_f, a_r = pp.scale_action(action, env_cfg)
        out = {
            "dos": dos,
            "doa": doa,
            "pred_dos": pred_dos,
            "pred_doa": pred_doa,
            "captures": info["captures"],
            "pred_reward": reward[:n0].mean() if n0 else nan,
            "prey_reward": reward[n0:].mean(),
            "prey_survival": info["survival"][n0:].mean(),
            "prey_movement": info["movement"][n0:].mean(),
            "pred_af": a_f[:n0].mean() if n0 else nan,
            "pred_ar": jnp.abs(a_r[:n0]).mean() if n0 else nan,
            "prey_af": a_f[n0:].mean(),
            "prey_ar": jnp.abs(a_r[n0:]).mean(),
            "pred_z": jnp.abs(z_pred).mean() if n0 else nan,
            "prey_z": jnp.abs(z_prey).mean(),
            "pred_q": p_aux["q"],
            "prey_q": y_aux["q"],
            "pred_critic_loss": p_aux["critic_loss"],
            "prey_critic_loss": y_aux["critic_loss"],
        }
        carry = carry._replace(
            pred=pred,
            prey=prey,
            state=state,
            obs=next_obs,
            key=key,
            step=carry.step + 1,
        )
        return carry, out

    # Episodic sums: the paper plots return per agent per episode.
    SUMMED = ("pred_reward", "prey_reward", "prey_survival", "prey_movement")

    def episode(carry, _):
        key, k_reset = jax.random.split(carry.key)
        state = pp.reset(k_reset, env_cfg)
        carry = carry._replace(key=key, state=state, obs=pp.observe(state, env_cfg))
        carry, out = jax.lax.scan(env_step, carry, None, length=env_cfg.episode_len)

        def decay(x):
            return jnp.maximum(tr.expl_min, x - tr.expl_decay)

        carry = carry._replace(eps=decay(carry.eps), noise=decay(carry.noise))
        summary = {
            name: (v.sum() if name in SUMMED else v.mean()) for name, v in out.items()
        }
        return carry, {**summary, "eps": carry.eps}

    def train(key):
        k_pred, k_prey, k_reset, key = jax.random.split(key, 4)
        state = pp.reset(k_reset, env_cfg)
        carry = Carry(
            pred=init(k_pred),
            prey=init(k_prey),
            state=state,
            obs=pp.observe(state, env_cfg),
            key=key,
            eps=jnp.float32(tr.eps),
            noise=jnp.float32(tr.noise),
            step=jnp.int32(0),
        )
        carry, history = jax.lax.scan(episode, carry, None, length=tr.episodes)
        return {
            "metrics": history,
            "pred_actor": carry.pred.actor,
            "prey_actor": carry.prey.actor,
        }

    return train
