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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("env tests passed")
