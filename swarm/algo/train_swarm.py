"""Two-species DDPG coevolution — the replication (Li 2023 algorithm 1).

An actor-critic, a replay buffer and a normaliser per species; conspecifics share
theirs. One scan over episodes, an inner scan over steps, buffers in the carry.
"""
import flax.struct as struct
import jax
import jax.numpy as jnp

from swarm.algo import ddpg
from swarm.envs import metrics, predator_prey as pp
from swarm.envs.scripted import scripted_predator


@struct.dataclass
class Carry:
    pred: ddpg.Agent
    prey: ddpg.Agent
    pred_buf: ddpg.Buffer
    prey_buf: ddpg.Buffer
    pred_norm: ddpg.ObsNorm
    prey_norm: ddpg.ObsNorm
    state: pp.EnvState
    obs: jnp.ndarray
    key: jnp.ndarray
    eps: jnp.float32
    noise: jnp.float32


def make_train(cfg, params):
    dim = pp.obs_dim(params)
    algo = ddpg.make_ddpg(cfg, dim, 2)
    n0 = params.n_pred
    pred, prey = slice(0, n0), slice(n0, None)
    periodic = params.boundary == "torus"
    zero = jnp.float32(0.0)
    no_aux = {"critic_loss": zero, "actor_loss": zero, "q_mean": zero}
    pred_cap = 1 if cfg.scripted_predator else cfg.buffer_size

    def species_metrics(pos, theta):
        if pos.shape[0] < 2:
            return jnp.nan, jnp.nan       # a lone agent has no nearest neighbour
        return (metrics.dos(pos, params.edge, periodic),
                metrics.doa(pos, theta, params.edge, periodic))

    def step(c, _):
        key, k_pred, k_prey, s_pred, s_prey = jax.random.split(c.key, 5)
        a_pred = (scripted_predator(c.state, params) if cfg.scripted_predator else
                  algo.act(c.pred, ddpg.normalize(c.pred_norm, c.obs[pred]), k_pred, c.eps, c.noise))
        a_prey = algo.act(c.prey, ddpg.normalize(c.prey_norm, c.obs[prey]), k_prey, c.eps, c.noise)

        obs_d, state, reward, _, info = pp.step(
            c.state, jnp.concatenate([a_pred, a_prey]), params)
        nobs = pp.flatten_obs(obs_d)

        def learn(agent, buf, norm, sl, action, k):
            buf = ddpg.insert(buf, c.obs[sl], action, reward[sl], nobs[sl])
            if cfg.normalize_obs:
                norm = ddpg.update_obsnorm(norm, nobs[sl])

            def do(_):
                o, a, r, no = ddpg.sample(buf, k, cfg.batch_size)
                return algo.update(agent, (ddpg.normalize(norm, o), a, r,
                                           ddpg.normalize(norm, no)))

            def skip(_):
                return agent, no_aux

            agent, aux = jax.lax.cond(buf.size >= cfg.learning_starts, do, skip, None)
            return agent, buf, norm, aux

        if cfg.scripted_predator:
            p_agent, p_buf, p_norm, p_aux = c.pred, c.pred_buf, c.pred_norm, no_aux
        else:
            p_agent, p_buf, p_norm, p_aux = learn(c.pred, c.pred_buf, c.pred_norm,
                                                 pred, a_pred, s_pred)
        y_agent, y_buf, y_norm, y_aux = learn(c.prey, c.prey_buf, c.prey_norm, prey, a_prey, s_prey)

        dos, doa = species_metrics(state.pos[prey], state.theta[prey])
        pred_dos, pred_doa = species_metrics(state.pos[pred], state.theta[pred])
        out = {"dos": dos, "doa": doa, "pred_dos": pred_dos, "pred_doa": pred_doa,
               "captures": info["captures"],
               "pred_reward": reward[pred].mean() if n0 else jnp.nan,
               "prey_reward": reward[prey].mean(),
               "prey_survival": info["survival"][prey].mean(),
               "prey_movement": info["movement"][prey].mean(),
               "prey_critic_loss": y_aux["critic_loss"], "prey_q": y_aux["q_mean"],
               "pred_critic_loss": p_aux["critic_loss"], "pred_q": p_aux["q_mean"]}

        c = c.replace(pred=p_agent, prey=y_agent, pred_buf=p_buf, prey_buf=y_buf,
                      pred_norm=p_norm, prey_norm=y_norm, state=state, obs=nobs, key=key)
        return c, out

    def episode(c, _):
        key, k_reset = jax.random.split(c.key)
        obs_d, state = pp.reset(k_reset, params)
        c = c.replace(key=key, state=state, obs=pp.flatten_obs(obs_d))
        c, out = jax.lax.scan(step, c, None, length=params.episode_len)

        decay = lambda x: jnp.maximum(cfg.expl_min, x - cfg.expl_decay)
        c = c.replace(eps=decay(c.eps), noise=decay(c.noise))

        summed = ("pred_reward", "prey_reward", "prey_survival", "prey_movement")
        m = {k: (v.sum() if k in summed else v.mean()) for k, v in out.items()}
        return c, {**m, "eps": c.eps}

    def train(key):
        k_pred, k_prey, k_reset, key = jax.random.split(key, 4)
        obs_d, state = pp.reset(k_reset, params)
        c = Carry(pred=algo.init(k_pred), prey=algo.init(k_prey),
                  pred_buf=ddpg.new_buffer(pred_cap, dim, 2),
                  prey_buf=ddpg.new_buffer(cfg.buffer_size, dim, 2),
                  pred_norm=ddpg.new_obsnorm(dim), prey_norm=ddpg.new_obsnorm(dim),
                  state=state, obs=pp.flatten_obs(obs_d), key=key,
                  eps=jnp.float32(cfg.eps), noise=jnp.float32(cfg.noise))
        c, m = jax.lax.scan(episode, c, None, length=cfg.episodes)
        return {"pred": c.pred, "prey": c.prey,
                "pred_norm": c.pred_norm, "prey_norm": c.prey_norm, "metrics": m}

    return train
