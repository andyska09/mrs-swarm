# Implementation choices

The other half of `research/notes/li2023_spec.md`. That file is the paper side
only; this one records what *we* decided where the paper does not decide for us,
and where we hardcode something it does.

Rule, in three categories:

1. **The paper pins it down** → hardcode. A knob whose value is already known is
   just a way to get it wrong later.
2. **The paper leaves it open** → config field, so the choice is recorded in
   every run directory rather than buried in the implementation.
3. **Derived from other config** → computed in code, never a field. If it were a
   field you could set it inconsistently with the values it follows from.

---

## Hardcoded

| Choice | Value | Why |
|---|---|---|
| Speed limit | explicit clamp after integration | Table 1 gives one drag (2) and one max acc (1), so the drag equilibrium is `max_acc/drag = 0.5` — exactly the tabulated max speed, which is why appendix B never mentions enforcing it. But §4.5 needs 0.3, and drag cannot produce that without per-species coefficients that contradict table 1. A clamp is the only consistent reading. Without it `speed_prey: 0.3` is a dead field and §4.5 silently re-runs the 1:1 baseline. |
| Multiple simultaneous contacts | score once, not once per adversary | §3.4 says "the reward for a predator is `r = +1` if it catches prey" — a binary condition, not a count. So a predator touching three prey still gets +1, and a prey touched by two predators still gets −1. Each of those three prey does bleed its own −1. |
| Spawn overlap | rejection-sample until every pair is ≥ `r_i + r_j` apart | Agents may touch but never start inside each other. Otherwise Hooke at stiffness 50 on mass 1 fires a large impulse on step 1, and it worsens as radii grow. Spec §7: minimum separation never stated. |


## Config fields — the paper is silent

| Field | Spec | Choice and why |
|---|---|---|
| `radius_pred` / `radius_prey` | §7 | Never stated, and capture *is* contact, so these two numbers set the entire capture rate and therefore the survival-vs-movement reward balance. |
| `heading_encoding` | §7 | `unit` = `[cos θ, sin θ]`, giving `d_o = 54`; `angle` gives 41. Unit avoids the ±π seam, where two nearly identical headings sit at opposite ends of the input range. |
| `init_speed_frac` | §7 | Spawn speed as a fraction of *that agent's own* max, random direction. A fraction rather than m/s so the 5:3 runs cannot spawn the slow species above its own cap. |
| `actor_output` | §3.2, §7 | `tanh` on both dimensions, then a per-dimension affine rescale: `a_F = (x+1)/2 · max_acc`, `a_R = x · max_ang_vel`. The paper states only that outputs are "re-scaled to fit within specified ranges" and that the networks use ReLU — but ReLU on the output cannot produce a negative `a_R`, so a bounded output is forced without being named. |
| `optimizer` | §7 | `adam`. Never stated; it is the MADDPG reference default, and table 2's learning rates only mean something relative to an optimizer. |
| where the action enters the critic | §3.2, §7 | **Not** a config field: concatenated with the observation at the input layer, as in the MADDPG reference. It is a code path, not a number, and nothing in the paper suggests varying it. One line to make it a knob if an ablation ever needs one. |
| `learning_starts` | §4 | Buffer warm-up, never stated. Counted in **env steps, not transitions** — each step inserts `n_prey` prey rows but only `n_pred` predator rows, so a transition threshold would silently start the two species learning at different times. |

## Config fields — the paper fixes the value, we keep the knob

`catch_reward`, `cost_af`, `cost_ar`, `boundary_penalty` are all stated in §3.4
(±1, −0.01·|a_F|, −0.1·|a_R|, −0.1). They stay configurable anyway, because the
reward balance is the most likely thing to need an ablation: the survival term is
worth ~1 unit per prey-episode against a movement budget of ~6, and whether
evasion pays at all turns on that ratio.

## Derived, never configured

`D`, the maximum possible distance between two agents:

```
torus:  D = edge · √2 / 2      minimum image caps per-axis separation at edge/2
walls:  D = edge · √2          full diagonal, no wrap
```

Eq (2) states the periodic case explicitly ("for a periodic square environment
with edge length 2, the largest possible distance is √2"). It has two uses and
they must not drift apart:

- perception range, `R = perception_frac · D` (§4.6: `R = D`, `2/3 D`, `1/3 D`)
- the DoS normalisation in eq (2)

## Run storage

- `configs/` — hand-written, complete JSONs. Missing *and* unknown keys are both
  errors, so nothing is inherited from the code. The only thing you edit.
- `runs/<timestamp>_<hash8>/` — append-only pool. Holds `config.json`,
  `meta.json`, and one `s<seed>/` folder per seed. Never named by hand, so a
  grid search needs no naming scheme.
- `experiments/<name>/` — a list of run ids and nothing else. An experiment is a
  *view* over the pool, so the runs in it need not share a config: that is how a
  sweep is represented.

The hash covers `algo` + `env` + `model` + `train`. Not the name, not the seeds,
not the commit — so "same hyperparameters, different seeds, run a month apart"
stays one experiment, and adding a seed later adds a run rather than forking the
experiment. Because the hash is *in* the run id, a stale entry in an experiment
list is detectable with no extra bookkeeping.

Duplicate `(hash, seed)` resolves to the newest. The commit sha is recorded in
`meta.json`; there is no dirty-tree check and no completion marker.
