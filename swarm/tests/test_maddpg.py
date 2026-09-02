"""MADDPG tests. Spec 3 and 4. Short runs — this is a gate, not an experiment.

    python -m swarm.tests.test_maddpg
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from swarm.algo import maddpg
from swarm.config import load

ROOT = Path(__file__).resolve().parents[2]
FULL = load(ROOT / "configs" / "flocking.json")

# 5e5 x 2 species is 440 MB of buffer; a gate does not need it.
SMALL = replace(
    FULL,
    train=replace(
        FULL.train, episodes=40, buffer_size=20_000, batch_size=64, learning_starts=100
    ),
)


def _run(exp, seed=0):
    out = jax.jit(maddpg.make_train(exp))(jax.random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in out["metrics"].items()}, out


def test_metrics_have_one_row_per_episode():
    m, _ = _run(SMALL)
    for name, v in m.items():
        assert v.shape == (SMALL.train.episodes,), name


def test_everything_stays_finite():
    m, _ = _run(SMALL)
    for name, v in m.items():
        if name.startswith("pred_") and SMALL.env.n_pred < 2 and "dos" in name:
            continue
        assert np.all(np.isfinite(v)), name


def test_the_run_is_deterministic_in_its_seed():
    a, _ = _run(SMALL, seed=0)
    b, _ = _run(SMALL, seed=0)
    c, _ = _run(SMALL, seed=1)
    assert np.array_equal(a["prey_reward"], b["prey_reward"])
    assert not np.array_equal(a["prey_reward"], c["prey_reward"])


def test_no_gradient_step_before_learning_starts():
    """Warm-up is counted in env steps, so both species start together."""
    exp = replace(SMALL, train=replace(SMALL.train, learning_starts=250))
    m, _ = _run(exp)
    warm_episodes = exp.train.learning_starts // exp.env.episode_len
    assert np.all(m["prey_critic_loss"][:warm_episodes] == 0.0)
    assert np.all(m["pred_critic_loss"][:warm_episodes] == 0.0)
    assert m["prey_critic_loss"][-1] != 0.0
    assert m["pred_critic_loss"][-1] != 0.0


def test_exploration_decays_per_episode():
    """max(0.05, eps - 5e-5) once per episode, not once per step. Spec 3.5."""
    m, _ = _run(SMALL)
    tr = SMALL.train
    assert abs(m["eps"][0] - (tr.eps - tr.expl_decay)) < 1e-9
    assert np.all(np.diff(m["eps"]) <= 0.0)
    assert np.all(m["eps"] >= tr.expl_min)


def test_critics_learn_the_sign_of_their_own_reward():
    """Contact pays a predator +1 and a prey -1, so the two Q values must split."""
    m, _ = _run(SMALL)
    assert m["pred_q"][-1] > 0.0 > m["prey_q"][-1]


def test_checkpoint_holds_both_actors_and_no_buffer():
    _, out = _run(SMALL)
    assert set(out) == {"metrics", "pred_actor", "prey_actor"}
    leaves = jax.tree.leaves(out["pred_actor"])
    assert leaves and all(np.all(np.isfinite(x)) for x in leaves)


def test_actors_actually_move_off_their_initialisation():
    exp = replace(SMALL, train=replace(SMALL.train, episodes=10))
    _, trained = _run(exp)
    fresh = jax.jit(maddpg.make_train(replace(exp, train=replace(exp.train, episodes=0))))
    start = fresh(jax.random.PRNGKey(0))
    for a, b in zip(
        jax.tree.leaves(start["prey_actor"]), jax.tree.leaves(trained["prey_actor"])
    ):
        if a.size > 1:
            assert not np.array_equal(a, b)
            break


def test_zero_predators_leaves_the_prey_alone():
    """Spec C25: the n0 = 0 control. No captures, and no predator update."""
    exp = replace(SMALL, env=replace(SMALL.env, n_pred=0))
    m, _ = _run(exp)
    assert np.all(m["captures"] == 0.0)
    assert np.all(m["pred_critic_loss"] == 0.0)
    assert np.all(np.isfinite(m["dos"])) and np.all(np.isfinite(m["doa"]))


def test_parallel_envs_keep_the_metric_shape():
    """n_envs changes how much data a step collects, not the shape of anything."""
    par = replace(SMALL, train=replace(SMALL.train, n_envs=8))
    m, _ = _run(par)
    one, _ = _run(SMALL)
    for name, v in m.items():
        assert v.shape == (par.train.episodes,), name
        assert np.all(np.isfinite(v)), name
    assert not np.allclose(m["dos"], one["dos"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("maddpg tests passed")
