"""DoS and DoA tests. Spec section 5.

    python -m swarm_simple.tests.test_metrics
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from swarm_simple.config import load
from swarm_simple.envs import metrics
from swarm_simple.envs import predator_prey as pp

ROOT = Path(__file__).resolve().parents[2]
CFG = load(ROOT / "configs" / "flocking.json").env


def _uniform(key, n, cfg):
    half = cfg.edge / 2
    return jax.random.uniform(key, (n, 2), minval=-half, maxval=half)


def test_doa_of_random_headings_is_two_over_pi():
    """The paper computes this itself: E[cos(phi/2)] = 2/pi ~ 0.64 (spec C4)."""
    pos = _uniform(jax.random.PRNGKey(0), 4000, CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(1), (4000,), minval=-jnp.pi, maxval=jnp.pi)
    assert abs(float(metrics.doa(pos, theta, CFG)) - 2 / jnp.pi) < 0.02


def test_doa_is_one_when_aligned():
    pos = _uniform(jax.random.PRNGKey(0), 200, CFG)
    assert abs(float(metrics.doa(pos, jnp.zeros(200), CFG)) - 1.0) < 1e-5


def test_doa_is_zero_for_anti_aligned_pairs():
    """Tight pairs, so each agent's nearest neighbour is its flipped partner."""
    x = jnp.repeat(jnp.linspace(-0.9, 0.9, 20), 2)
    y = jnp.tile(jnp.array([0.0, 0.001]), 20)
    pos = jnp.stack([x, y], axis=-1)
    theta = jnp.tile(jnp.array([0.0, jnp.pi]), 20)
    assert float(metrics.doa(pos, theta, CFG)) < 1e-5


def test_doa_is_local_not_global_polarisation():
    """Two tight flocks pointing opposite ways: DoA ~ 1, group mean would read 0.

    Spec C15 — this is why the paper defines both measures over a neighbour.
    """
    left = jnp.stack([jnp.linspace(-0.9, -0.6, 50), jnp.zeros(50)], axis=-1)
    right = jnp.stack([jnp.linspace(0.6, 0.9, 50), jnp.zeros(50)], axis=-1)
    pos = jnp.concatenate([left, right])
    theta = jnp.concatenate([jnp.zeros(50), jnp.full(50, jnp.pi)])
    assert float(metrics.doa(pos, theta, replace(CFG, boundary="walls"))) > 0.99


def test_dos_is_zero_when_coincident():
    assert float(metrics.dos(jnp.zeros((10, 2)), CFG)) == 0.0


def test_dos_stays_in_the_unit_interval():
    for i in range(20):
        d = float(metrics.dos(_uniform(jax.random.PRNGKey(i), 10, CFG), CFG))
        assert 0.0 < d < 1.0


def test_dos_of_uniform_prey_matches_the_papers_starting_value():
    """10 uniform prey on the edge-2 torus give ~0.226, the paper's 22% (spec C2).

    This is the check on the D normalisation: dividing by edge = 2 instead of
    D = sqrt(2) would read 0.16 and every DoS number would be incomparable.
    """
    got = jnp.mean(
        jnp.stack(
            [metrics.dos(_uniform(jax.random.PRNGKey(i), CFG.n_prey, CFG), CFG)
             for i in range(400)]
        )
    )
    assert abs(float(got) - 0.226) < 0.01


def test_dos_normalisation_follows_the_boundary():
    """Identical positions, different D. Walls divide by the full diagonal, so the
    same configuration reads exactly half. Clustered in the middle, so no pair
    wraps and the raw nearest-neighbour distances are the same either way."""
    pos = jax.random.uniform(jax.random.PRNGKey(0), (10, 2), minval=-0.4, maxval=0.4)
    walls = replace(CFG, boundary="walls")
    assert abs(pp.max_distance(walls) / pp.max_distance(CFG) - 2.0) < 1e-12
    assert abs(float(metrics.dos(pos, walls)) * 2 - float(metrics.dos(pos, CFG))) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("metric tests passed")
