"""Replay buffer tests. Spec 3.3.

    python -m swarm_simple.tests.test_buffer
"""

import jax
import jax.numpy as jnp

from swarm_simple.algo import buffer

D_O, D_A = 4, 2
KEY = jax.random.PRNGKey(0)


def _rows(values):
    """One transition per value, every field tagged with it so rows are traceable."""
    v = jnp.asarray(values, dtype=jnp.float32)[:, None]
    return (
        jnp.tile(v, (1, D_O)),
        jnp.tile(v, (1, D_A)),
        v[:, 0],
        jnp.tile(-v, (1, D_O)),
    )


def test_empty_shapes():
    buf = buffer.empty(16, D_O, D_A)
    assert buf.obs.shape == (16, D_O)
    assert buf.action.shape == (16, D_A)
    assert buf.reward.shape == (16,)
    assert buf.next_obs.shape == (16, D_O)
    assert int(buf.ptr) == 0 and int(buf.size) == 0


def test_insert_writes_one_row_per_conspecific():
    """A step inserts n_i rows at once, not one. Spec 3.3."""
    buf = buffer.insert(buffer.empty(16, D_O, D_A), *_rows([1, 2, 3]))
    assert int(buf.size) == 3 and int(buf.ptr) == 3
    assert jnp.array_equal(buf.reward[:3], jnp.array([1.0, 2.0, 3.0]))
    assert float(buf.reward[3]) == 0.0


def test_size_saturates_and_pointer_wraps():
    buf = buffer.empty(5, D_O, D_A)
    buf = buffer.insert(buf, *_rows([1, 2, 3]))
    buf = buffer.insert(buf, *_rows([4, 5, 6]))
    assert int(buf.size) == 5  # capacity, not 6
    assert int(buf.ptr) == 1  # (3 + 3) % 5
    # Rows land at 3, 4, 0 — so the oldest row is the one overwritten.
    assert jnp.array_equal(buf.reward, jnp.array([6.0, 2.0, 3.0, 4.0, 5.0]))


def test_insert_straddling_the_end_loses_nothing():
    """n need not divide capacity: 3 predators into a buffer sized for 10 prey."""
    buf = buffer.empty(10, D_O, D_A)
    for start in range(0, 30, 3):
        buf = buffer.insert(buf, *_rows([start + 1, start + 2, start + 3]))
    assert int(buf.size) == 10
    assert sorted(int(x) for x in buf.reward) == list(range(21, 31))


def test_sample_shapes():
    buf = buffer.insert(buffer.empty(16, D_O, D_A), *_rows([1, 2, 3]))
    obs, action, reward, next_obs = buffer.sample(buf, KEY, 8)
    assert obs.shape == (8, D_O)
    assert action.shape == (8, D_A)
    assert reward.shape == (8,)
    assert next_obs.shape == (8, D_O)


def test_sample_never_returns_unwritten_rows():
    """4 of 64 slots filled: the zeros must never be drawn."""
    buf = buffer.insert(buffer.empty(64, D_O, D_A), *_rows([1, 2, 3, 4]))
    reward = buffer.sample(buf, KEY, 2000)[2]
    assert set(int(x) for x in reward) <= {1, 2, 3, 4}


def test_a_sampled_row_stays_one_transition():
    """obs, action, reward and next_obs must be drawn at the same index."""
    buf = buffer.insert(buffer.empty(64, D_O, D_A), *_rows([1, 2, 3, 4]))
    obs, action, reward, next_obs = buffer.sample(buf, KEY, 500)
    assert jnp.allclose(obs[:, 0], reward)
    assert jnp.allclose(action[:, 0], reward)
    assert jnp.allclose(next_obs[:, 0], -reward)


def test_dtypes_survive_insert():
    """A lax.scan carry breaks if ptr or size changes dtype mid-loop."""
    a = buffer.empty(16, D_O, D_A)
    b = buffer.insert(a, *_rows([1, 2, 3]))
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape


def test_insert_and_sample_under_jit():
    ins = jax.jit(buffer.insert)
    buf = ins(buffer.empty(16, D_O, D_A), *_rows([1, 2, 3]))
    assert int(buf.size) == 3
    assert jax.jit(buffer.sample, static_argnums=2)(buf, KEY, 8)[2].shape == (8,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("buffer tests passed")
