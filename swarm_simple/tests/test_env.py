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


def _blocks(obs, cfg):
    """Flat observation -> (own, predator slots, prey slots)."""
    h, k = pp.heading_dim(cfg), cfg.n_neighbors
    own, rest = obs[..., : 4 + h], obs[..., 4 + h :]
    rest = rest.reshape(*obs.shape[:-1], 2 * k, 2 + h)
    return own, rest[..., :k, :], rest[..., k:, :]


def _visible(slots):
    return jnp.any(slots != 0.0, axis=-1)


def test_obs_dim_is_54_with_unit_headings():
    assert pp.obs_dim(CFG) == 54
    assert pp.obs_dim(replace(CFG, heading_encoding="angle")) == 41


def test_observe_shape():
    obs = pp.observe(RESET(KEYS[0], CFG), CFG)
    assert obs.shape == (pp.n_agents(CFG), pp.obs_dim(CFG))


def test_predators_come_before_prey_for_a_prey_observer():
    """Spec 1.6. The ordering is by species, not by ally/adversary."""
    cfg = replace(CFG, n_pred=1, n_prey=2)
    state = pp.State(
        pos=jnp.array([[0.1, 0.0], [0.0, 0.0], [0.2, 0.0]]),  # pred, observer, prey
        vel=jnp.zeros((3, 2)),
        theta=jnp.zeros(3),
        time=jnp.int32(0),
    )
    _, pred, prey = _blocks(pp.observe(state, cfg)[1], cfg)
    assert jnp.allclose(pred[0, :2], jnp.array([0.1, 0.0]))
    assert jnp.allclose(prey[0, :2], jnp.array([0.2, 0.0]))


def test_never_observes_itself():
    for k in KEYS[:20]:
        obs = pp.observe(RESET(k, CFG), CFG)
        _, pred, prey = _blocks(obs, CFG)
        for slots in (pred, prey):
            at_zero = jnp.all(slots[..., :2] == 0.0, axis=-1)
            assert not bool(jnp.any(at_zero & _visible(slots)))


def test_neighbours_are_nearest_first():
    for k in KEYS[:20]:
        _, pred, prey = _blocks(pp.observe(RESET(k, CFG), CFG), CFG)
        for slots in (pred, prey):
            d = jnp.linalg.norm(slots[..., :2], axis=-1)
            d = jnp.where(_visible(slots), d, 1e6)  # finite: inf - inf is nan
            assert jnp.all(jnp.diff(d, axis=-1) >= -1e-6)


def test_perception_radius_masks_and_pads_with_zeros():
    near = replace(CFG, perception_frac=0.02)
    state = RESET(KEYS[0], CFG)
    wide = _visible(_blocks(pp.observe(state, CFG), CFG)[2]).sum()
    tight = _visible(_blocks(pp.observe(state, near), near)[2]).sum()
    assert int(tight) < int(wide)
    _, pred, prey = _blocks(pp.observe(state, near), near)
    for slots in (pred, prey):
        assert float(jnp.abs(slots[~_visible(slots)]).max()) == 0.0


def test_topological_limit_caps_visible_neighbours():
    """20 prey all inside R, but a prey may still only see n_neighbors of them."""
    crowd = replace(CFG, n_prey=20)
    _, pred, prey = _blocks(pp.observe(RESET(KEYS[0], crowd), crowd), crowd)
    assert int(_visible(prey)[crowd.n_pred :].sum(-1).max()) == crowd.n_neighbors
    assert int(_visible(pred).sum(-1).max()) == crowd.n_pred


def test_short_block_masks_out_rather_than_wrapping():
    """3 predators into 6 slots: the last 3 are zeros, not a repeat of the first."""
    _, pred, _ = _blocks(pp.observe(RESET(KEYS[0], CFG), CFG), CFG)
    assert int(_visible(pred).sum(-1).max()) <= CFG.n_pred
    assert float(jnp.abs(pred[:, CFG.n_pred :]).max()) == 0.0


def _pair(cfg, gap):
    """One predator and one prey `gap` metres apart, at rest."""
    return pp.State(
        pos=jnp.array([[0.0, 0.0], [gap, 0.0]]),
        vel=jnp.zeros((2, 2)),
        theta=jnp.zeros(2),
        time=jnp.int32(0),
    )


ONE_V_ONE = replace(CFG, n_pred=1, n_prey=1)
LAZY = jnp.array([[-1.0, 0.0], [-1.0, 0.0]])  # a_F = 0, a_R = 0, so no movement cost


def test_contact_pays_plus_one_and_minus_one():
    touching = CFG.radius_pred + CFG.radius_prey - 1e-3
    r, info = pp.reward(_pair(ONE_V_ONE, touching), LAZY, ONE_V_ONE)
    assert jnp.allclose(r, jnp.array([1.0, -1.0]))
    assert jnp.allclose(info["survival"], jnp.array([1.0, -1.0]))
    assert float(info["captures"]) == 1.0


def test_separated_agents_score_zero():
    apart = CFG.radius_pred + CFG.radius_prey + 1e-3
    r, info = pp.reward(_pair(ONE_V_ONE, apart), LAZY, ONE_V_ONE)
    assert jnp.allclose(r, 0.0)
    assert float(info["captures"]) == 0.0


def test_touching_exactly_is_not_a_capture():
    """Same threshold as contact_force: contact scores, mere touching does not."""
    exact = CFG.radius_pred + CFG.radius_prey
    _, info = pp.reward(_pair(ONE_V_ONE, exact), LAZY, ONE_V_ONE)
    assert float(info["captures"]) == 0.0


def test_reward_is_paid_every_step_the_contact_persists():
    """Spec 2: prey are not removed, so it is not once per capture event."""
    state = _pair(ONE_V_ONE, CFG.radius_pred + CFG.radius_prey - 1e-3)
    paid = []
    for _ in range(5):
        paid.append(float(pp.reward(state, LAZY, ONE_V_ONE)[0][0]))
        state = state._replace(time=state.time + 1)
    assert paid == [1.0] * 5


def test_movement_cost_matches_spec_2():
    n = pp.n_agents(CFG)
    action = jnp.tile(jnp.array([[1.0, 1.0]]), (n, 1))  # a_F = max_acc, a_R = +max
    _, info = pp.reward(pp.reset(KEYS[0], CFG), action, CFG)
    want = -(CFG.cost_af * CFG.max_acc + CFG.cost_ar * CFG.max_ang_vel)
    assert jnp.allclose(info["movement"], want)


def test_doing_nothing_costs_nothing():
    n = pp.n_agents(CFG)
    _, info = pp.reward(pp.reset(KEYS[0], CFG), jnp.tile(LAZY[:1], (n, 1)), CFG)
    assert jnp.allclose(info["movement"], 0.0)


def test_one_predator_touching_two_prey_still_scores_once():
    cfg = replace(CFG, n_pred=1, n_prey=2)
    d = CFG.radius_pred + CFG.radius_prey - 1e-3
    state = pp.State(
        pos=jnp.array([[0.0, 0.0], [d, 0.0], [-d, 0.0]]),
        vel=jnp.zeros((3, 2)),
        theta=jnp.zeros(3),
        time=jnp.int32(0),
    )
    r, info = pp.reward(state, jnp.tile(LAZY[:1], (3, 1)), cfg)
    assert float(r[0]) == 1.0
    assert float(info["captures"]) == 2.0  # both prey bleed


def test_no_wall_penalty_on_a_torus():
    for k in KEYS[:20]:
        _, info = pp.reward(RESET(k, CFG), jnp.zeros((pp.n_agents(CFG), 2)), CFG)
        assert float(jnp.abs(info["wall"]).max()) == 0.0


def test_wall_penalty_fires_only_against_a_boundary():
    walls = replace(CFG, boundary="walls", n_pred=1, n_prey=1)
    half = walls.edge / 2
    state = pp.State(
        pos=jnp.array([[half, 0.0], [0.0, 0.0]]),  # first is jammed into the wall
        vel=jnp.zeros((2, 2)),
        theta=jnp.zeros(2),
        time=jnp.int32(0),
    )
    _, info = pp.reward(state, LAZY, walls)
    assert jnp.allclose(info["wall"], jnp.array([-walls.boundary_penalty, 0.0]))


def _random_policy(key, obs):
    return jax.random.uniform(key, (obs.shape[0], 2), minval=-1.0, maxval=1.0)


ROLLOUT = jax.jit(pp.rollout, static_argnums=(1, 2))


def test_rollout_shapes_and_length():
    states, rewards, info = ROLLOUT(KEYS[0], CFG, _random_policy)
    t, n = CFG.episode_len, pp.n_agents(CFG)
    assert states.pos.shape == (t, n, 2)
    assert states.theta.shape == (t, n)
    assert rewards.shape == (t, n)
    assert info["captures"].shape == (t,)
    assert int(states.time[-1]) == t


def test_rollout_is_deterministic_in_key_and_config():
    a = ROLLOUT(KEYS[0], CFG, _random_policy)[1]
    b = ROLLOUT(KEYS[0], CFG, _random_policy)[1]
    c = ROLLOUT(KEYS[1], CFG, _random_policy)[1]
    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)


def test_long_rollout_stays_finite_and_bounded():
    long = replace(CFG, episode_len=2000)
    states, rewards, _ = jax.jit(pp.rollout, static_argnums=(1, 2))(
        KEYS[0], long, _random_policy
    )
    assert jnp.all(jnp.isfinite(states.pos)) and jnp.all(jnp.isfinite(rewards))
    assert jnp.all(jnp.abs(states.pos) <= long.edge / 2 + 1e-6)
    speed = jnp.linalg.norm(states.vel, axis=-1)
    assert float(speed.max()) <= float(pp.max_speeds(long).max()) + 1e-6


def test_the_policy_is_handed_observations_not_state():
    """A policy taking (key, obs) cannot reach around the perception limit."""
    seen = []

    def spy(key, obs):
        seen.append(obs.shape)
        return jnp.zeros((obs.shape[0], 2))

    pp.rollout(KEYS[0], replace(CFG, episode_len=2), spy)
    assert seen == [(pp.n_agents(CFG), pp.obs_dim(CFG))]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("env tests passed")
