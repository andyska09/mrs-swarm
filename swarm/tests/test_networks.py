"""Actor and critic tests. Spec 3.2.

    python -m swarm.tests.test_networks
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from swarm.algo import networks
from swarm.config import load
from swarm.envs import predator_prey as pp

ROOT = Path(__file__).resolve().parents[2]
CFG = load(ROOT / "configs" / "flocking.json")
D_O, D_A = pp.obs_dim(CFG.env), pp.act_dim(CFG.env)
ACTOR, CRITIC = networks.build(CFG.model, D_A)
KEY = jax.random.PRNGKey(0)


def _init():
    obs = jnp.zeros((1, D_O))
    act = jnp.zeros((1, D_A))
    return ACTOR.init(KEY, obs), CRITIC.init(KEY, obs, act)


def _obs(n):
    return jax.random.normal(jax.random.PRNGKey(1), (n, D_O))


def test_three_hidden_layers_of_64():
    """Table 2: 3 hidden layers, 64 wide. Plus one output head each."""
    a_params, c_params = _init()
    for params, head in ((a_params, D_A), (c_params, 1)):
        dense = params["params"]
        assert len(dense) == len(CFG.model.hidden) + 1
        widths = [dense[f"Dense_{i}"]["kernel"].shape[1] for i in range(len(dense))]
        assert widths == list(CFG.model.hidden) + [head]


def test_critic_takes_the_action_at_the_input_layer():
    _, c_params = _init()
    assert c_params["params"]["Dense_0"]["kernel"].shape[0] == D_O + D_A


def test_actor_output_shape_and_range():
    a_params, _ = _init()
    a, z = ACTOR.apply(a_params, _obs(32))
    assert a.shape == z.shape == (32, D_A)
    assert jnp.all(jnp.abs(a) <= 1.0)


def test_critic_returns_a_scalar_per_row():
    _, c_params = _init()
    q = CRITIC.apply(c_params, _obs(32), jnp.zeros((32, D_A)))
    assert q.shape == (32,)


def test_networks_broadcast_over_leading_axes():
    """The training loop applies one actor to (n_envs, n_agents, d_o)."""
    a_params, c_params = _init()
    obs = jax.random.normal(KEY, (4, 10, D_O))
    assert ACTOR.apply(a_params, obs)[0].shape == (4, 10, D_A)
    assert CRITIC.apply(c_params, obs, jnp.zeros((4, 10, D_A))).shape == (4, 10)


def test_actor_is_deterministic_and_not_collapsed():
    a_params, _ = _init()
    obs = _obs(64)
    assert jnp.array_equal(ACTOR.apply(a_params, obs)[0], ACTOR.apply(a_params, obs)[0])
    assert float(ACTOR.apply(a_params, obs)[0].std()) > 1e-4


def test_actor_output_feeds_scale_action_directly():
    """The net emits [-1, 1]; the env owns the rescale into (a_F, a_R)."""
    a_params, _ = _init()
    n = pp.n_agents(CFG.env)
    a_f, a_r = pp.scale_action(ACTOR.apply(a_params, _obs(n))[0], CFG.env)
    assert jnp.all((a_f >= 0.0) & (a_f <= CFG.env.max_acc))
    assert jnp.all(jnp.abs(a_r) <= CFG.env.max_ang_vel)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("network tests passed")
