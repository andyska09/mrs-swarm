"""
PureJaxRL-style PPO for continuous control. Full-jit: the whole training loop
(rollout, GAE, epochs, minibatches, num_updates iterations) compiles into ONE
XLA program. Call `jax.jit(make_train(config, env, env_params))(rng)`.

Env-agnostic: any gymnax env with `obs_size`, `num_actions`, and — if you want
truncation bootstrapping — info["terminal_obs"], info["terminated"],
info["truncated"]. Outcome metrics (capture/miss) are read from info if the
env provides r_capture / r_miss, else skipped (Python-level check, static).

Features (all optional; defaults match PureJaxRL unless noted):
- separate actor/critic MLPs
- truncation bootstrapping (Pardo 2018)                    [on by default here]
- value-loss clipping (off), post-normalization clipping (off)
- diagnostics: clipfrac, approx_kl, entropy, explained variance

References: Huang et al. "37 Implementation Details of PPO" (2022);
Andrychowicz et al. "What Matters in On-Policy RL" (2021);
Pardo et al. "Time Limits in RL" (2018); CleanRL; PureJaxRL.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import flax.struct as struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax

from ppo.config import TrainConfig
from ppo.wrappers import (LogWrapper, ClipAction, VecEnv, NormalizeVecObservation,
                          NormalizeVecReward, TerminalObsWrapper)


@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


class PPOBatch(NamedTuple):
    """Slim batch for the epoch/minibatch scan — no info dict to shuffle."""
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


class ActorCritic(nn.Module):
    action_dim: int
    actor_dims: tuple = (256, 256)
    critic_dims: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.tanh if self.activation == "tanh" else nn.relu

        a = x
        for d in self.actor_dims:
            a = act(nn.Dense(d, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(a))
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

        c = x
        for d in self.critic_dims:
            c = act(nn.Dense(d, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(c))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return pi, jnp.squeeze(value, axis=-1)


def build_network(config: TrainConfig, action_dim: int) -> ActorCritic:
    return ActorCritic(action_dim=action_dim, actor_dims=config.actor_dims,
                       critic_dims=config.effective_critic_dims,
                       activation=config.activation)


def make_train(config: TrainConfig, env, env_params):
    """Returns train(rng) -> {"runner_state", "metrics"}. Jit it yourself."""
    obs_size, action_dim = env.obs_size, env.num_actions

    # ── Wrapper chain: TerminalObs -> Log -> Clip -> Vec -> NormObs -> NormRew ──
    # Log sits INSIDE normalization so logged returns are raw env rewards.
    if config.truncation_bootstrap:
        env = TerminalObsWrapper(env)
    env = LogWrapper(env)
    if config.clip_action:
        env = ClipAction(env)
    env = VecEnv(env)
    if config.normalize_obs:
        env = NormalizeVecObservation(env, clip_obs=config.clip_obs, clip_range=config.clip_obs_range)
    if config.normalize_reward:
        env = NormalizeVecReward(env, config.gamma, clip_reward=config.clip_reward,
                                 clip_range=config.clip_reward_range)

    num_updates, num_envs, num_steps = config.num_updates, config.num_envs, config.num_steps
    minibatch_size = config.minibatch_size
    assert num_updates > 0, "total_timesteps < num_envs * num_steps"
    assert (num_envs * num_steps) % config.num_minibatches == 0

    def train(rng):
        network = build_network(config, action_dim)
        rng, k = jax.random.split(rng)
        params = network.init(k, jnp.zeros(obs_size))

        if config.anneal_lr:
            def schedule(count):
                frac = 1.0 - (count // (config.update_epochs * config.num_minibatches)) / num_updates
                return config.lr * frac
            lr = schedule
        else:
            lr = config.lr
        tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                         optax.adam(learning_rate=lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, k = jax.random.split(rng)
        obsv, env_state = env.reset(jax.random.split(k, num_envs), env_params)

        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # ── Rollout ──
            def _env_step(rs, _):
                train_state, env_state, last_obs, rng = rs
                rng, k_act, k_step = jax.random.split(rng, 3)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=k_act)
                log_prob = pi.log_prob(action)
                obsv, env_state, reward, done, info = env.step(
                    jax.random.split(k_step, num_envs), env_state, action, env_params)
                tr = Transition(done, action, value, reward, log_prob, last_obs, info)
                return (train_state, env_state, obsv, rng), tr

            (train_state, env_state, last_obs, rng), traj = lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, length=num_steps)

            # ── Truncation bootstrap: inject gamma*V(s_T) on timeout-only steps ──
            if config.truncation_bootstrap:
                _, v_term = network.apply(train_state.params, traj.info["terminal_obs"])
                trunc_only = (traj.info["truncated"].astype(jnp.float32)
                              * (1.0 - traj.info["terminated"].astype(jnp.float32)))
                traj = traj.replace(reward=traj.reward + config.gamma * v_term * trunc_only)

            # ── Episode metrics ──
            m = traj.info["returned_episode"]
            n_ep = m.sum()
            def _mean(x):
                return jnp.where(n_ep > 0, (x * m).sum() / n_ep, jnp.nan)
            metric = {
                "mean_return": _mean(traj.info["returned_episode_returns"]),
                "mean_ep_length": _mean(traj.info["returned_episode_lengths"]),
                "completed_episodes": n_ep,
            }
            if "r_capture" in traj.info:   # env-specific outcomes (static check)
                cap = (traj.info["r_capture"] > 0).astype(jnp.float32)
                miss = (traj.info["r_miss"] < 0).astype(jnp.float32)
                metric["capture_rate"] = jnp.where(n_ep > 0, cap.sum() / n_ep, jnp.nan)
                metric["miss_rate"] = jnp.where(n_ep > 0, miss.sum() / n_ep, jnp.nan)

            # ── GAE ──
            _, last_val = network.apply(train_state.params, last_obs)

            def _gae(carry, tr):
                gae, next_v = carry
                delta = tr.reward + config.gamma * next_v * (1 - tr.done) - tr.value
                gae = delta + config.gamma * config.gae_lambda * (1 - tr.done) * gae
                return (gae, tr.value), gae

            _, adv = lax.scan(_gae, (jnp.zeros_like(last_val), last_val), traj,
                              reverse=True, unroll=16)
            targets = adv + traj.value
            batch = PPOBatch(traj.obs, traj.action, traj.log_prob, traj.value)

            # ── PPO epochs ──
            def _epoch(us, _):
                train_state, batch, adv, targets, rng = us
                rng, k = jax.random.split(rng)
                B = num_envs * num_steps
                perm = jax.random.permutation(k, B)
                flat = jax.tree.map(lambda x: x.reshape((B,) + x.shape[2:]), batch)
                sh = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), flat)
                sh_adv = jnp.take(adv.reshape(B), perm)
                sh_tgt = jnp.take(targets.reshape(B), perm)
                mbs = (jax.tree.map(lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]), sh),
                       sh_adv.reshape(config.num_minibatches, minibatch_size),
                       sh_tgt.reshape(config.num_minibatches, minibatch_size))

                def _minibatch(train_state, mb):
                    b, a_mb, t_mb = mb

                    def _loss(params):
                        pi, value = network.apply(params, b.obs)
                        log_prob = pi.log_prob(b.action)
                        entropy = pi.entropy().mean()
                        ratio = jnp.exp(log_prob - b.log_prob)
                        a_n = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                        actor_loss = -jnp.minimum(
                            ratio * a_n,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * a_n).mean()
                        if config.clip_vloss:
                            v_clip = b.value + jnp.clip(value - b.value, -config.clip_eps, config.clip_eps)
                            value_loss = 0.5 * jnp.maximum((value - t_mb) ** 2, (v_clip - t_mb) ** 2).mean()
                        else:
                            value_loss = 0.5 * ((value - t_mb) ** 2).mean()
                        total = actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy
                        return total, (value_loss, actor_loss, entropy, ratio)

                    (total, (vl, al, ent, ratio)), grads = jax.value_and_grad(_loss, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                    approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio + 1e-8))
                    return train_state, (total, vl, al, ent, clipfrac, approx_kl)

                train_state, li = lax.scan(_minibatch, train_state, mbs)
                return (train_state, batch, adv, targets, rng), li

            (train_state, _, _, _, rng), li = lax.scan(
                _epoch, (train_state, batch, adv, targets, rng), None, length=config.update_epochs)

            y_pred, y_true = traj.value.reshape(-1), targets.reshape(-1)
            var_y = jnp.var(y_true)
            metric.update({
                "value_loss": li[1][-1].mean(), "actor_loss": li[2][-1].mean(),
                "entropy": li[3][-1].mean(), "clipfrac": li[4][-1].mean(),
                "approx_kl": li[5][-1].mean(),
                "explained_var": jnp.where(var_y > 0, 1.0 - jnp.var(y_true - y_pred) / var_y, jnp.nan),
            })
            return (train_state, env_state, last_obs, rng), metric

        rng, k = jax.random.split(rng)
        runner_state, metrics = lax.scan(_update_step, (train_state, env_state, obsv, k),
                                         None, length=num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
