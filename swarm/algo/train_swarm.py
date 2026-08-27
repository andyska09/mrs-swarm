"""Two-species DDPG coevolution — the replication (Li 2023 algorithm 1).

An actor-critic, a replay buffer and a normaliser per species; conspecifics share
theirs. One scan over episodes, an inner scan over steps, buffers in the carry.

`cfg.freeze_period` alternates which species learns. Who learns is a Python bool
baked in at trace time, never a lax.cond: vmap turns a cond with a batched
predicate into a select that runs both branches.
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


def _flat(tree):
    """Merge a scan's two leading axes: (outer, inner, ...) -> (outer*inner, ...)."""
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), tree)


def make_train(cfg, params):
    dim = pp.obs_dim(params)
    algo = ddpg.make_ddpg(cfg, dim, 2)
    n0, n_env = params.n_pred, cfg.n_envs
    pred, prey = (slice(None), slice(0, n0)), (slice(None), slice(n0, None))   # env axis first
    periodic = params.boundary == "torus"
    zero = jnp.float32(0.0)
    no_aux = {"critic_loss": zero, "actor_loss": zero, "q_mean": zero}
    pred_cap = 1 if cfg.scripted_predator else cfg.buffer_size

    reset_v = jax.vmap(lambda k: pp.reset(k, params))
    step_v = jax.vmap(lambda s, a: pp.step(s, a, params))
    scripted_v = jax.vmap(lambda s: scripted_predator(s, params))

    def species_metrics(pos, theta):
        if pos.shape[1] < 2:
            return jnp.nan, jnp.nan       # a lone agent has no nearest neighbour
        dos = jax.vmap(lambda p: metrics.dos(p, params.edge, periodic))(pos)
        doa = jax.vmap(lambda p, t: metrics.doa(p, t, params.edge, periodic))(pos, theta)
        return dos.mean(), doa.mean()

    def make_step(do_pred, do_prey):
        def step(c, _):
            key, k_pred, k_prey, s_pred, s_prey = jax.random.split(c.key, 5)
            a_pred = (scripted_v(c.state) if cfg.scripted_predator else
                      algo.act(c.pred, ddpg.normalize(c.pred_norm, c.obs[pred]), k_pred,
                               c.eps, c.noise))
            a_prey = algo.act(c.prey, ddpg.normalize(c.prey_norm, c.obs[prey]), k_prey,
                              c.eps, c.noise)

            obs_d, state, reward, _, info = step_v(
                c.state, jnp.concatenate([a_pred, a_prey], axis=1))
            nobs = pp.flatten_obs(obs_d)

            def learn(agent, buf, norm, sl, action, k, train):
                o, no = c.obs[sl].reshape(-1, dim), nobs[sl].reshape(-1, dim)
                buf = ddpg.insert(buf, o, action.reshape(-1, 2), reward[sl].reshape(-1), no)
                if cfg.normalize_obs:
                    norm = ddpg.update_obsnorm(norm, no)
                if not train:
                    return agent, buf, norm, no_aux   # frozen: collects, never updates

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
                                                     pred, a_pred, s_pred, do_pred)
            y_agent, y_buf, y_norm, y_aux = learn(c.prey, c.prey_buf, c.prey_norm,
                                                  prey, a_prey, s_prey, do_prey)

            dos, doa = species_metrics(state.pos[prey], state.theta[prey])
            pred_dos, pred_doa = species_metrics(state.pos[pred], state.theta[pred])
            out = {"dos": dos, "doa": doa, "pred_dos": pred_dos, "pred_doa": pred_doa,
                   "captures": info["captures"].mean(),
                   "pred_reward": reward[pred].mean() if n0 else jnp.nan,
                   "prey_reward": reward[prey].mean(),
                   "prey_survival": info["survival"][prey].mean(),
                   "prey_movement": info["movement"][prey].mean(),
                   "prey_af": info["a_f"][prey].mean(), "prey_ar": info["a_r"][prey].mean(),
                   "pred_af": info["a_f"][pred].mean() if n0 else jnp.nan,
                   "pred_ar": info["a_r"][pred].mean() if n0 else jnp.nan,
                   "prey_critic_loss": y_aux["critic_loss"],
                   "prey_actor_loss": y_aux["actor_loss"], "prey_q": y_aux["q_mean"],
                   "pred_critic_loss": p_aux["critic_loss"],
                   "pred_actor_loss": p_aux["actor_loss"], "pred_q": p_aux["q_mean"]}

            c = c.replace(pred=p_agent, prey=y_agent, pred_buf=p_buf, prey_buf=y_buf,
                          pred_norm=p_norm, prey_norm=y_norm, state=state, obs=nobs, key=key)
            return c, out
        return step

    def make_episode(do_pred, do_prey):
        step = make_step(do_pred, do_prey)
        phase = {"pred_learning": jnp.float32(do_pred and not cfg.scripted_predator),
                 "prey_learning": jnp.float32(do_prey)}

        def episode(c, _):
            key, k_reset = jax.random.split(c.key)
            obs_d, state = reset_v(jax.random.split(k_reset, n_env))
            c = c.replace(key=key, state=state, obs=pp.flatten_obs(obs_d))
            c, out = jax.lax.scan(step, c, None, length=params.episode_len)

            decay = lambda x: jnp.maximum(cfg.expl_min, x - cfg.expl_decay)
            c = c.replace(eps=decay(c.eps), noise=decay(c.noise))

            summed = ("pred_reward", "prey_reward", "prey_survival", "prey_movement")
            m = {k: (v.sum() if k in summed else v.mean()) for k, v in out.items()}
            return c, {**m, "eps": c.eps, **phase}
        return episode

    n_chunk = cfg.episodes // cfg.n_ckpt
    if cfg.freeze_period:
        ep_pred, ep_prey = make_episode(True, False), make_episode(False, True)

        def pair(c, _):
            c, m_pred = jax.lax.scan(ep_pred, c, None, length=cfg.freeze_period)
            c, m_prey = jax.lax.scan(ep_prey, c, None, length=cfg.freeze_period)
            return c, jax.tree.map(lambda a, b: jnp.concatenate([a, b]), m_pred, m_prey)

        def episodes(c):
            c, m = jax.lax.scan(pair, c, None, length=n_chunk // (2 * cfg.freeze_period))
            return c, _flat(m)
    else:
        ep_both = make_episode(True, True)

        def episodes(c):
            return jax.lax.scan(ep_both, c, None, length=n_chunk)

    def chunk(c, _):
        c, m = episodes(c)
        return c, ({"pred": c.pred.actor, "prey": c.prey.actor,
                    "pred_norm": c.pred_norm, "prey_norm": c.prey_norm}, m)

    def train(key):
        k_pred, k_prey, k_reset, key = jax.random.split(key, 4)
        obs_d, state = reset_v(jax.random.split(k_reset, n_env))
        c = Carry(pred=algo.init(k_pred), prey=algo.init(k_prey),
                  pred_buf=ddpg.new_buffer(pred_cap, dim, 2),
                  prey_buf=ddpg.new_buffer(cfg.buffer_size, dim, 2),
                  pred_norm=ddpg.new_obsnorm(dim), prey_norm=ddpg.new_obsnorm(dim),
                  state=state, obs=pp.flatten_obs(obs_d), key=key,
                  eps=jnp.float32(cfg.eps), noise=jnp.float32(cfg.noise))
        c, (ckpts, m) = jax.lax.scan(chunk, c, None, length=cfg.n_ckpt)
        return {"pred": c.pred, "prey": c.prey, "ckpts": ckpts,
                "pred_norm": c.pred_norm, "prey_norm": c.prey_norm, "metrics": _flat(m)}

    return train
