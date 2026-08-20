# Writing your own environment

The learner needs a gymnax-style env with a handful of properties. Copy
`interceptor2d.py`, replace the physics, keep the skeleton. The contract:

```python
class MyEnv(gymnax.environments.environment.Environment):
    obs_size = ...          # int, static
    num_actions = ...       # int, static (continuous action dim)

    def reset_env(self, key, params) -> (obs, state)
    def step_env(self, key, state, action, params) -> (obs, state, reward, done, info)
    def get_obs(self, state, params) -> obs
    def is_terminal(self, state, params) -> bool   # TRUE terminals only
    def action_space(self, params) / observation_space(self, params)
```

`Environment.step` (inherited) wraps `step_env` and **auto-resets on `done`**.
That's convenient for training and a trap everywhere else — see README know-how #4.

## Rules that keep it jit-able and correct

**State is a `flax.struct.dataclass`.** All fields are arrays or scalars. Same
dtypes out of `reset_env` and `step_env` (`jnp.int32(0)` vs `0` is a bug that
surfaces as a cryptic `lax.scan` carry error). No Python objects, no lists.

**Params too — and static ones are declared static.** Anything you branch on in
Python must be `struct.field(pytree_node=False)`. Then `if params.mode == ...`
is resolved at trace time and each variant compiles its own step. Anything that
varies per-episode or per-env goes in *state*, drawn in `reset_env` from `key`.

**Traced values never meet Python control flow.** `jnp.where`, `lax.cond`,
`lax.select`. Fixed-size everything: no dynamic shapes, no growing lists — if
you need a buffer, pre-allocate and mask.

**`done = terminated | truncated`, and put both in `info`.** Capture/crash/miss
are terminals; timeout is a truncation. The trainer bootstraps `V(s_T)` on
truncation-only steps and needs three keys in `info`:

```python
info = {
    "terminated": terminated,          # bool
    "truncated":  truncated,           # bool
    "terminal_obs": self.get_obs(new_state, params),   # PRE-reset obs, every step
    ...
}
```

`terminal_obs` is computed inside `step_env`, before the inherited `step` does
its reset select — that is the only place the true last observation exists.

**Reward is a pure function of (state, next_state, action, cfg).** Keep it in
the same file until it grows. Dense shaping (like the progress term here) makes
learning fast; keep it simple and make sure the sparse terminal bonuses still
dominate what the agent is actually rewarded for.

**Optional outcome keys.** If `info` contains `r_capture` / `r_miss` (or you
rename the check in `ppo/train.py`), the trainer logs capture/miss rates for free.

**Presets are the only place values are chosen.** A `PRESETS` dict of
`EnvParams` at the bottom of the file, `get_env_params(name)`. Nothing
hard-coded in `run/`.

## Checklist before training

- `python tests/test_smoke.py` passes with your env swapped in (edit the two
  import lines). Add one assertion per thing your env can get wrong.
- Random-policy rollout: rewards finite, episodes terminate, obs magnitude
  sane (or you rely on `normalize_obs=True`).
- The learning gate goes up on the easiest preset. If it doesn't in a few
  million steps, the reward is wrong before the algorithm is.
