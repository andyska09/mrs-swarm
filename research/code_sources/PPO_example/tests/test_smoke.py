"""Smoke tests. Run before any GPU job:  python tests/test_smoke.py

Env gates (seconds):   shapes, jit, vmap, no NaN, termination, dtype
                       consistency reset vs step, truncation semantics.
Learning gate (~10 s on CPU): a short PPO run must raise mean return and
                       capture rate. Catches "it compiles but doesn't learn".
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from envs import Interceptor2D, get_env_params
from ppo.config import TrainConfig
from ppo.train import make_train


def ok(msg):
    print(f"  ✓ {msg}")


def env_gates():
    print("ENV GATES")
    env, p = Interceptor2D(), get_env_params("straight")
    key = jax.random.PRNGKey(0)

    obs, st = env.reset(key, p)
    assert obs.shape == (env.obs_size,)
    ok(f"reset obs {obs.shape}")

    key, k = jax.random.split(key)
    a = jax.random.uniform(k, (env.num_actions,), minval=-1, maxval=1)
    obs2, st2, r, d, info = env.step(key, st, a, p)
    assert obs2.shape == obs.shape and r.shape == () and d.shape == ()
    ok("step shapes")

    jax.jit(env.step)(key, st, a, p)
    ok("jit(step)")

    N = 64
    keys = jax.random.split(key, N)
    ob, sb = jax.vmap(env.reset, in_axes=(0, None))(keys, p)
    acts = jax.random.uniform(key, (N, env.num_actions), minval=-1, maxval=1)
    vstep = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    ob2, _, rb, db, _ = vstep(keys, sb, acts, p)
    assert ob2.shape == (N, env.obs_size) and rb.shape == (N,)
    ok(f"vmap({N})")

    def _scan(c, _):
        st, key = c
        key, k1, k2 = jax.random.split(key, 3)
        a = jax.random.uniform(k1, (env.num_actions,), minval=-1, maxval=1)
        obs, st, r, d, info = env.step(k2, st, a, p)
        return (st, key), (r, d, obs)
    _, (rs, ds, obss) = jax.lax.scan(_scan, (st, key), None, length=1000)
    assert jnp.all(jnp.isfinite(rs)) and jnp.all(jnp.isfinite(obss))
    assert jnp.any(ds), "no termination in 1000 random steps"
    ok(f"1000 random steps: finite, terminates (reward range [{rs.min():.2f}, {rs.max():.2f}])")

    _, sr = env.reset_env(key, p)
    _, ss, _, _, _ = env.step_env(key, sr, a, p)
    for f in sr.__dataclass_fields__:
        rv, sv = getattr(sr, f), getattr(ss, f)
        assert getattr(rv, "dtype", type(rv)) == getattr(sv, "dtype", type(sv)), f"dtype drift in {f}"
    ok("state dtypes stable reset->step (else lax.scan carry breaks)")

    st_late = sr.replace(time=jnp.int32(p.max_steps_in_episode - 1))
    _, _, _, d_t, info_t = env.step(key, st_late, jnp.zeros(2), p)
    assert bool(d_t) and bool(info_t["truncated"]) and not bool(info_t["terminated"])
    assert "terminal_obs" in info_t
    ok("timeout -> truncated (not terminated); info['terminal_obs'] present")


def learning_gate():
    print("LEARNING GATE (short CPU run)")
    cfg = TrainConfig(num_envs=256, num_steps=64, num_minibatches=8,
                      total_timesteps=256 * 64 * 60, seed=0)   # ~1M steps, 60 updates
    env, p = Interceptor2D(), get_env_params("straight")
    train = jax.jit(make_train(cfg, env, p))
    t0 = time.time()
    out = train(jax.random.PRNGKey(0))
    jax.block_until_ready(out)
    wall = time.time() - t0
    m = {k: np.asarray(v) for k, v in out["metrics"].items()}
    for k in ("mean_return", "value_loss", "entropy", "explained_var"):
        assert np.isfinite(m[k][-1]), f"{k} not finite"
    early, late = np.nanmean(m["mean_return"][:10]), np.nanmean(m["mean_return"][-10:])
    cap_e, cap_l = np.nanmean(m["capture_rate"][:10]), np.nanmean(m["capture_rate"][-10:])
    print(f"  {cfg.total_timesteps/1e6:.1f}M steps in {wall:.0f}s "
          f"({cfg.total_timesteps/wall/1e3:.0f}k sps) | return {early:.1f} -> {late:.1f} | "
          f"capture {cap_e:.2f} -> {cap_l:.2f}")
    assert late > early + 5.0, "return did not improve"
    assert cap_l > cap_e + 0.2, "capture rate did not improve"
    ok("PPO learns on 'straight' (return up, capture rate up)")


if __name__ == "__main__":
    env_gates()
    learning_gate()
    print("\n✅ ALL SMOKE TESTS PASSED")
