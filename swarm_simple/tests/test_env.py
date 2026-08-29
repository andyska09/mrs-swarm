"""Environment tests. Spec section 1, choices.md for what the paper leaves open.

    python -m swarm_simple.tests.test_env
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from swarm_simple.config import load
from swarm_simple.envs import predator_prey as pp

ROOT = Path(__file__).resolve().parents[2]
CFG = load(ROOT / "configs" / "flocking.json").env
RESET = jax.jit(pp.reset, static_argnums=1)
STEP = jax.jit(pp.step, static_argnums=2)
KEYS = [jax.random.PRNGKey(i) for i in range(100)]


def _min_gap(state, cfg):
    """Smallest (distance - contact distance) over all pairs. Negative = overlap."""
    n, r = pp.n_agents(cfg), pp.radii(cfg)
    d = jnp.linalg.norm(pp.delta(state.pos[:, None, :], state.pos[None, :, :], cfg), axis=-1)
    return float((d - (r[:, None] + r[None, :]) + jnp.eye(n) * 10).min())


def test_max_distance_matches_eq2():
    assert abs(pp.max_distance(CFG) - 2**0.5) < 1e-12
    walls = replace(CFG, boundary="walls")
    assert abs(pp.max_distance(walls) - 2 * 2**0.5) < 1e-12


def test_perception_is_a_fraction_of_d():
    assert abs(pp.perception(CFG) - pp.max_distance(CFG)) < 1e-12
    third = replace(CFG, perception_frac=1 / 3)
    assert abs(pp.perception(third) - pp.max_distance(CFG) / 3) < 1e-12


def test_minimum_image_wraps_on_torus():
    a, b = jnp.array([-0.99, 0.0]), jnp.array([0.99, 0.0])
    assert abs(float(jnp.linalg.norm(pp.delta(a, b, CFG))) - 0.02) < 1e-6


def test_minimum_image_does_not_wrap_with_walls():
    a, b = jnp.array([-0.99, 0.0]), jnp.array([0.99, 0.0])
    walls = replace(CFG, boundary="walls")
    assert abs(float(jnp.linalg.norm(pp.delta(a, b, walls))) - 1.98) < 1e-6


def test_reset_shapes():
    s = RESET(KEYS[0], CFG)
    n = pp.n_agents(CFG)
    assert s.pos.shape == (n, 2)
    assert s.vel.shape == (n, 2)
    assert s.theta.shape == (n,)
    assert s.time.shape == ()


def test_spawn_inside_the_arena():
    half = CFG.edge / 2
    for k in KEYS:
        assert jnp.all(jnp.abs(RESET(k, CFG).pos) <= half + 1e-6)


def test_spawn_headings_wrapped():
    for k in KEYS:
        assert jnp.all(jnp.abs(RESET(k, CFG).theta) <= jnp.pi)


def test_spawn_speed_within_cap():
    cap = pp.max_speeds(CFG) * CFG.init_speed_frac
    for k in KEYS:
        speed = jnp.linalg.norm(RESET(k, CFG).vel, axis=-1)
        assert jnp.all(speed <= cap + 1e-6)


def test_agents_never_spawn_inside_each_other():
    for k in KEYS:
        assert _min_gap(RESET(k, CFG), CFG) >= 0.0


def test_no_overlap_at_ablation_radii():
    """The 3x radii of the radius ablation are where a naive spawn breaks."""
    dense = replace(CFG, radius_pred=0.18, radius_prey=0.12)
    reset = jax.jit(pp.reset, static_argnums=1)
    for k in KEYS:
        assert _min_gap(reset(k, dense), dense) >= 0.0


def _rollout(cfg, action, steps, key=KEYS[0]):
    state = pp.reset(key, cfg)
    trace = []
    for _ in range(steps):
        state = STEP(state, action, cfg)
        trace.append(state)
    return state, trace


def test_action_scaling_matches_spec_1_3():
    a = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    a_f, a_r = pp.scale_action(a, CFG)
    assert jnp.allclose(a_f, jnp.array([0.0, 0.5, 1.0]) * CFG.max_acc)
    assert jnp.allclose(a_r, jnp.array([-1.0, 0.0, 1.0]) * CFG.max_ang_vel)


def test_action_is_clipped():
    a_f, a_r = pp.scale_action(jnp.array([[9.0, -9.0]]), CFG)
    assert float(a_f[0]) == CFG.max_acc and float(a_r[0]) == -CFG.max_ang_vel


def test_position_uses_the_pre_update_velocity():
    """Spec 1.2: x(t+1) = x(t) + v(t)dt. From rest, one step cannot move anything."""
    rest = pp.reset(KEYS[0], replace(CFG, init_speed_frac=0.0))
    moved = STEP(rest, jnp.ones((pp.n_agents(CFG), 2)), replace(CFG, init_speed_frac=0.0))
    assert jnp.allclose(moved.pos, rest.pos)
    assert float(jnp.linalg.norm(moved.vel, axis=-1).max()) > 0.0


def test_drag_equilibrium_is_max_acc_over_drag():
    """One agent, clamp lifted: full throttle settles at a_F/drag = 0.5 unaided.

    That equilibrium equals table 1's max speed, which is why appendix B never
    mentions enforcing a limit. Alone, so no contact force pollutes it.
    """
    solo = replace(
        CFG, n_pred=1, n_prey=0, speed_pred=100.0, init_speed_frac=0.0
    )
    state, _ = _rollout(solo, jnp.array([[1.0, 0.0]]), 300)
    speed = float(jnp.linalg.norm(state.vel, axis=-1).max())
    assert abs(speed - solo.max_acc / solo.drag) < 1e-3


def test_speed_clamp_binds_for_the_slow_species():
    """Without the clamp, speed_prey 0.3 would be a dead field and 4.5 would be a no-op."""
    ratio = replace(CFG, speed_prey=0.3)
    state, trace = _rollout(ratio, jnp.ones((pp.n_agents(ratio), 2)), 200)
    prey = jnp.stack([jnp.linalg.norm(s.vel[ratio.n_pred :], axis=-1) for s in trace])
    pred = jnp.stack([jnp.linalg.norm(s.vel[: ratio.n_pred], axis=-1) for s in trace])
    assert float(prey.max()) <= 0.3 + 1e-6
    assert float(pred.max()) > 0.3


def test_no_tunnelling_through_contact():
    """One step of travel must be shorter than the contact distance."""
    _, trace = _rollout(CFG, jnp.ones((pp.n_agents(CFG), 2)), 200)
    hops = [
        float(jnp.linalg.norm(pp.delta(a.pos, b.pos, CFG), axis=-1).max())
        for a, b in zip(trace, trace[1:])
    ]
    assert max(hops) < CFG.radius_pred + CFG.radius_prey


def test_stays_on_the_torus_and_headings_stay_wrapped():
    _, trace = _rollout(CFG, jnp.ones((pp.n_agents(CFG), 2)), 300)
    for s in trace:
        assert jnp.all(jnp.abs(s.pos) <= CFG.edge / 2 + 1e-6)
        assert jnp.all(jnp.abs(s.theta) <= jnp.pi + 1e-6)


def test_overlapping_agents_push_apart_without_diverging():
    """Tightly clustered, not coincident: coincident is degenerate, see contact_force."""
    n = pp.n_agents(CFG)
    ang = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    pos = 0.01 * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)
    state = pp.State(pos, jnp.zeros((n, 2)), jnp.zeros(n), jnp.int32(0))
    assert _min_gap(state, CFG) < 0.0  # they really do start overlapped

    for _ in range(300):
        state = STEP(state, jnp.full((n, 2), -1.0), CFG)
    assert jnp.all(jnp.isfinite(state.pos))
    assert _min_gap(state, CFG) > -1e-3


def test_walls_contain_and_report_contact():
    walls = replace(CFG, boundary="walls")
    _, trace = _rollout(walls, jnp.ones((pp.n_agents(walls), 2)), 300)
    touched = sum(int(pp.wall_force(s.pos, walls)[1].sum()) for s in trace)
    assert touched > 0
    for s in trace:
        assert jnp.all(jnp.abs(s.pos) <= walls.edge / 2 + 0.1)


def test_dtypes_survive_reset_to_step():
    """A lax.scan carry breaks if step returns a different dtype than reset."""
    a = pp.reset(KEYS[0], CFG)
    b = STEP(a, jnp.zeros((pp.n_agents(CFG), 2)), CFG)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("env tests passed")
