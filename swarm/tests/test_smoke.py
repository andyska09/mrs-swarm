"""Smoke tests. Green before any change is called done.

    python -m swarm.tests.test_smoke            # everything
    python -m swarm.tests.test_smoke --env      # env gates only (seconds)

METRIC GATES   DoA of random headings = 2/pi, DoS bounds, nearest-neighbour
               semantics.
ENV GATES      gate 1 of the plan: shapes, jit, dtype stability, torus wrap,
               minimum image, contacts resolve, no tunnelling, obs masking,
               and the population edge cases (n_pred=0, 50 prey).
LEARNING GATE  a short coevolution run: both critics must learn the sign of
               their own reward, and every metric must stay finite.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from swarm.algo.config import get_train_config
from swarm.envs import metrics, predator_prey as pp
from swarm.envs.scripted import all_random, scripted_predator, scripted_vs_random


def ok(msg):
    print(f"  ✓ {msg}")


# ── metrics ──────────────────────────────────────────────────────────────────

def metric_gates():
    print("METRIC GATES")
    key = jax.random.PRNGKey(0)
    n, edge = 2000, 2.0
    pos = jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)
    theta = jax.random.uniform(jax.random.PRNGKey(1), (n,), minval=-jnp.pi, maxval=jnp.pi)

    # Independent-of-position control: headings are random, so the nearest
    # neighbour is an arbitrary agent and E[||h_j + h_k||/2] = E[cos(phi/2)] = 2/pi.
    a = float(metrics.doa(pos, theta, edge, periodic=True))
    assert abs(a - 2 / jnp.pi) < 0.02, f"DoA of random headings = {a}, expected 2/pi = 0.637"
    ok(f"DoA(random headings) = {a:.4f} ~ 2/pi = {2 / jnp.pi:.4f}")

    assert abs(float(metrics.doa(pos, jnp.zeros(n), edge, True)) - 1.0) < 1e-5
    # Anti-aligned needs tight pairs, so that each agent's nearest neighbour is
    # the partner whose heading was flipped.
    x = jnp.repeat(jnp.linspace(-0.9, 0.9, 20), 2)
    y = jnp.tile(jnp.array([0.0, 0.001]), 20)
    paired = jnp.stack([x, y], -1)
    opposed = metrics.doa(paired, jnp.tile(jnp.array([0.0, jnp.pi]), 20), edge, True)
    assert float(opposed) < 1e-5, float(opposed)
    ok("DoA = 1 when aligned, 0 for anti-aligned nearest pairs")

    # Nearest-neighbour semantics: two tight flocks pointing opposite ways have
    # DoA ~ 1, where group-mean polarisation reads 0.
    left = jnp.stack([jnp.linspace(-0.9, -0.6, 50), jnp.zeros(50)], -1)
    right = jnp.stack([jnp.linspace(0.6, 0.9, 50), jnp.zeros(50)], -1)
    two = jnp.concatenate([left, right])
    th2 = jnp.concatenate([jnp.zeros(50), jnp.full(50, jnp.pi)])
    assert float(metrics.doa(two, th2, edge, False)) > 0.99
    ok("DoA of two opposed flocks ~ 1 (nearest neighbour, not group mean)")

    assert float(metrics.dos(jnp.zeros((10, 2)), edge, True)) == 0.0
    d_rand = float(metrics.dos(pos, edge, True))
    assert 0.0 < d_rand < 1.0
    ok(f"DoS in [0,1]: coincident = 0, uniform({n} agents) = {d_rand:.4f}")


# ── environment ──────────────────────────────────────────────────────────────

def env_gates():
    print("ENV GATES")
    params = pp.get_env_params("torus")
    key = jax.random.PRNGKey(0)
    n = pp.n_agents(params)

    obs, state = pp.reset(key, params)
    assert obs["self"].shape == (n, 6)
    assert obs["neighbors"].shape == (n, 2 * params.n_neighbors, 4)
    assert obs["mask"].shape == (n, 2 * params.n_neighbors)
    flat = pp.flatten_obs(obs)
    assert flat.shape == (n, pp.obs_dim(params)) == (n, 54)
    ok(f"reset: structured obs + flatten -> {flat.shape}")

    a = jax.random.uniform(key, (n, 2), minval=-1, maxval=1)
    obs2, state2, r, d, info = pp.step(state, a, params)
    assert r.shape == (n,) and d.shape == () and obs2["self"].shape == obs["self"].shape
    ok("step shapes")

    jax.jit(pp.step, static_argnums=2)(state, a, params)
    ok("jit(step)")

    for f in state.__dataclass_fields__:
        rv, sv = getattr(state, f), getattr(state2, f)
        assert rv.dtype == sv.dtype, f"dtype drift in {f}: {rv.dtype} -> {sv.dtype}"
    ok("state dtypes stable reset->step (else the lax.scan carry breaks)")

    # 1000 steps of random actions: finite, bounded, wrapped.
    long = params.replace(episode_len=1000)
    traj, rewards, info = jax.jit(pp.rollout, static_argnums=(1, 2))(key, long, all_random)
    assert jnp.all(jnp.isfinite(traj.pos)) and jnp.all(jnp.isfinite(traj.vel))
    half = params.edge / 2
    assert jnp.all(jnp.abs(traj.pos) <= half + 1e-5), "left the torus"
    assert jnp.all(jnp.abs(traj.theta) <= jnp.pi + 1e-5), "heading not wrapped"
    speed = jnp.linalg.norm(traj.vel, axis=-1)
    assert jnp.max(speed[:, params.n_pred:]) <= params.speed_prey + 1e-4
    assert jnp.max(speed[:, :params.n_pred]) <= params.speed_pred + 1e-4
    ok(f"1000 random steps: finite, wrapped, speed-capped (max {float(speed.max()):.3f})")

    # No tunnelling: one step of travel must be under the contact diameter.
    step_dist = jnp.linalg.norm((traj.pos[1:] - traj.pos[:-1] + half) % params.edge - half, axis=-1)
    contact = params.radius_pred + params.radius_prey
    assert float(step_dist.max()) < contact, f"{float(step_dist.max()):.4f} >= {contact}"
    ok(f"no tunnelling: max step {float(step_dist.max()):.4f} m < contact {contact} m")

    # Minimum image: two agents astride the seam are 0.02 m apart.
    seam = state.replace(pos=jnp.array([[-0.99, 0.0], [0.99, 0.0]] + [[0.0, 0.5]] * (n - 2)))
    _, dist = pp.pairwise(seam.pos, params)
    assert abs(float(dist[0, 1]) - 0.02) < 1e-5, float(dist[0, 1])
    ok(f"torus minimum image: agents at x=-0.99 and x=+0.99 are {float(dist[0, 1]):.3f} m apart")

    # Contacts push apart and settle at a finite separation.
    stuck = state.replace(pos=jnp.zeros((n, 2)), vel=jnp.zeros((n, 2)))
    s = stuck
    for _ in range(200):
        _, s, _, _, _ = pp.step(s, jnp.zeros((n, 2)), params)
    _, dfin = pp.pairwise(s.pos, params)
    dfin = dfin + jnp.eye(n) * 10
    r = pp.radii(params)
    assert jnp.all(jnp.isfinite(s.pos)), "collision blew up"
    assert float(dfin.min()) > 0.5 * float((r[:, None] + r[None, :]).min()), "never separated"
    ok(f"all {n} agents stacked on one point separate to {float(dfin.min()):.3f} m, no divergence")

    # Observation masking.
    assert not bool(jnp.any(jnp.all(obs["neighbors"][:, :, :2] == 0, -1) & obs["mask"])), \
        "a visible neighbour at zero relative position = an agent seeing itself"
    near = pp.get_obs(state, params.replace(perception=0.05))
    assert int(near["mask"].sum()) < int(obs["mask"].sum())
    assert float(jnp.abs(near["neighbors"][~near["mask"]]).max()) == 0.0
    ok("obs: self excluded, perception radius masks, masked slots are zero-padded")

    # Sorted nearest-first.
    d_neigh = jnp.linalg.norm(obs["neighbors"][:, :params.n_neighbors, :2], axis=-1)
    d_neigh = jnp.where(obs["mask"][:, :params.n_neighbors], d_neigh, 1e6)  # finite: inf-inf is nan
    assert jnp.all(jnp.diff(d_neigh, axis=-1) >= -1e-5), "neighbours not nearest-first"
    ok("neighbours sorted nearest-first")

    # Population edge cases. The obs dimension stays 54, which is what lets a
    # policy trained on 10 prey run on 50.
    for preset in ("npred0", "npred1", "eval50"):
        p = pp.get_env_params(preset)
        o, st = pp.reset(key, p)
        assert pp.flatten_obs(o).shape == (pp.n_agents(p), 54)
        traj, rew, _ = jax.jit(pp.rollout, static_argnums=(1, 2))(key, p, all_random)
        assert jnp.all(jnp.isfinite(traj.pos))
    ok("presets npred0 / npred1 / eval50 run; obs stays 54-dim")

    # Walls: agents stay inside the box and the boundary penalty fires.
    w = pp.get_env_params("walls")
    traj, rew, info = jax.jit(pp.rollout, static_argnums=(1, 2))(key, w, all_random)
    assert jnp.all(jnp.abs(traj.pos) < w.edge / 2 + 0.1), "escaped through a wall"
    assert float(info["wall_contacts"].sum()) > 0, "no wall contact in 100 random steps"
    ok(f"walls: contained, {float(info['wall_contacts'].sum()):.0f} wall contacts penalised")


def scripted_gate():
    print("SCRIPTED PREDATOR (the gate-2 baseline)")
    params = pp.get_env_params("torus")
    roll = jax.jit(pp.rollout, static_argnums=(1, 2))
    keys = jax.random.split(jax.random.PRNGKey(0), 16)

    # Long episodes: max angular velocity is 0.5 rad/s, so a predator needs ~6 s
    # to turn around. Over the paper's 100 steps (10 s) aiming barely pays off.
    long = params.replace(episode_len=300)
    scripted = np.array([float(roll(k, long, scripted_vs_random)[2]["captures"].mean())
                         for k in keys])
    random = np.array([float(roll(k, long, all_random)[2]["captures"].mean()) for k in keys])
    assert scripted.mean() > random.mean(), f"scripted {scripted.mean()} <= random {random.mean()}"
    ok(f"captures/step over 300 steps: scripted {scripted.mean():.3f} +- {scripted.std():.3f} "
       f"> random {random.mean():.3f} +- {random.std():.3f} (16 seeds)")

    # The rule itself, open loop: one predator, one reachable prey, the rest
    # parked far away. Open loop because a predator at full throttle overshoots
    # and its nearest prey changes underneath it.
    n = pp.n_agents(params)
    for bearing, want in ((jnp.pi / 2, 1.0), (-jnp.pi / 2, -1.0), (0.0, 0.0)):
        pos = jnp.full((n, 2), 0.9).at[0].set(jnp.array([0.0, 0.0]))
        pos = pos.at[params.n_pred].set(0.3 * jnp.array([jnp.cos(bearing), jnp.sin(bearing)]))
        st = pp.EnvState(pos=pos, vel=jnp.zeros((n, 2)), theta=jnp.zeros(n), time=jnp.int32(0))
        a = scripted_predator(st, params)
        assert float(a[0, 0]) == 1.0, "not full throttle"
        assert abs(float(a[0, 1]) - want) < 1e-4, f"bearing {bearing}: a_R = {float(a[0, 1])}"
    ok("rule: full throttle, a_R saturates toward the nearest prey and is 0 dead ahead")


# ── learner ──────────────────────────────────────────────────────────────────

def coevolution_gate(scripted=False):
    print(f"COEVOLUTION GATE ({'scripted' if scripted else 'learned'} predators)")
    from dataclasses import replace
    from swarm.algo.train_swarm import make_train

    cfg = replace(get_train_config("swirl" if scripted else "flocking"), episodes=80)
    params = pp.get_env_params(cfg.env_preset)
    t0 = time.time()
    out = jax.block_until_ready(jax.jit(make_train(cfg, params))(jax.random.PRNGKey(0)))
    wall = time.time() - t0

    m = {k: np.asarray(v) for k, v in out["metrics"].items()}
    for k, v in m.items():
        assert np.all(np.isfinite(v)), f"{k} not finite"
    print(f"  {cfg.total_steps} steps in {wall:.0f}s | captures "
          f"{m['captures'][:20].mean():.3f} -> {m['captures'][-20:].mean():.3f} | "
          f"q pred {m['pred_q'][-1]:+.2f} prey {m['prey_q'][-1]:+.2f}")
    # Contact pays a predator +1 and a prey -1, so the two Q values must split in sign.
    if scripted:
        assert m["pred_q"][-1] == 0.0, "a scripted predator must not train a critic"
    else:
        assert m["pred_q"][-1] > 0 > m["prey_q"][-1], "critics did not learn the reward sign"
    ok("prey learn; DoS/DoA logged and finite")


def freeze_gate():
    print("FREEZE GATE")
    from dataclasses import replace
    from swarm.algo.train_swarm import make_train

    cfg = replace(get_train_config("alt"), episodes=20, freeze_period=5, n_ckpt=2)
    params = pp.get_env_params(cfg.env_preset)
    out = jax.block_until_ready(jax.jit(make_train(cfg, params))(jax.random.PRNGKey(0)))
    m = {k: np.asarray(v) for k, v in out["metrics"].items()}

    assert m["pred_learning"][0] == 1.0 and m["prey_learning"][0] == 0.0, "predator must go first"
    for who in ("pred", "prey"):
        frozen = m[f"{who}_learning"] == 0.0
        assert frozen.any() and (~frozen).any(), f"{who} never alternates"
        assert np.all(m[f"{who}_actor_loss"][frozen] == 0.0), f"{who} updated while frozen"
    assert jax.tree.leaves(out["ckpts"])[0].shape[0] == cfg.n_ckpt
    ok(f"{cfg.freeze_period}-episode phases: a frozen species takes no gradient step; "
       f"{cfg.n_ckpt} checkpoints saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true", help="skip the learning gates")
    a = ap.parse_args()
    metric_gates()
    env_gates()
    scripted_gate()
    if not a.env:
        coevolution_gate()
        coevolution_gate(scripted=True)
        freeze_gate()
    print("\n✅ ALL SMOKE TESTS PASSED")
