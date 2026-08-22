# Plan — replicate Li 2023, then extend

Written to be read cold. Everything needed to start is here; the paper PDF is
only needed for figures.

Target: **Li, Li & Zhao (2023),** *Predator–prey survival pressure is sufficient
to evolve swarming behaviors*, New J. Phys. 25 092001.
Transcript: `research/papers/Li_2023_New_J._Phys._25_092001.md`.

## Purpose

Research internship, not a thesis. No claim to defend, no deadline, weekly
progress reports.

The deliverable is **a JAX predator–prey swarm platform that is cheap to extend**
toward 3D, attention, and realistic drone dynamics. Replicating the paper is how
the platform gets validated and how the MARL is learned. The replication is the
means; the platform is the end.

## Settled decisions, and why

**Algorithm: independent DDPG, not MADDPG.** The paper says "MADDPG with
modifications" and modification (1) is a *decentralised* critic — Algorithm 1
shows `Q_i(o_i, a_i)`, local observation and own action only. A centralised
critic is what the "MA" means; without it this is DDPG run per species with
parameter sharing across conspecifics and one shared replay buffer per species.
The only multi-agent content is that two species learn concurrently. Consequence:
no joint action space, and critic input dims do not grow with N — which is why a
policy trained on 10 prey deploys on 50. Name the file `ddpg.py`.

**Stack: JAX, own code, PPO_example's style.** No JaxMARL dependency. Read
JaxMARL (`MPE/simple_tag`, IPPO) and `quad-swarm-rl` (attention encoders) as
references, not as imports. The environment has to be written either way — Li's
env is not MPE.

**Faithful regime first.** The paper runs ONE environment: 2000 episodes × 100
steps = 200k env steps total, one update per step, batch 256 from a 5e5 buffer.
For off-policy learning the update-to-env-step ratio decides whether it learns at
all, so running thousands of parallel envs is a *different experiment*, not a
faster one. 200k steps is minutes on a laptop — `lax.scan` over steps with a
circular buffer in the carry keeps it jitted. Metacentrum/RCI are not needed for
the replication, only for phase 5.

**2D concretely.** No magic `2`, motion isolated from the first commit, but no
dimension-generic `D`. Heading does not generalise: 2D is a scalar θ with
`θ̇ = a_R`; the honest 3D version is a quadrotor with attitude, not a 3D unicycle.
A generic unicycle would be a fiction to delete later.

**Dynamics behind a seam.** `dynamics.py` exposes
`step_motion(state, action, params) -> state`, selected by a
`struct.field(pytree_node=False)` mode — the same pattern `interceptor2d.py` uses
for `evader_mode`, so each variant compiles its own step with no runtime branch.
This is a file boundary and one static field, not an abstraction layer. It is
what makes 3D and drone dynamics new functions instead of rewrites.

**Structured observation, paper-exact flattening.** The env emits
`{self, neighbors[K, F], mask[K]}`; a flatten encoder concatenates in the paper's
order to produce a bit-identical input to their MLP. Zero fidelity cost, and
attention later becomes a second encoder over the same env output rather than a
rewrite of the observation pipeline, the network, and every checkpoint. The mask
attention needs is the zero-padding the paper already specifies.

**Metrics are diagnostics, never rewards.** DoS/DoA live as pure functions in
`envs/metrics.py`. The paper's entire point is that these are not optimised;
keeping them structurally unable to reach the reward function is how that stays
true.

## Environment spec (Li 2023, §2 and appendix B)

2D continuous square, edge length 2 m. Two boundary modes:
- **periodic** (torus) — the main flocking result
- **walls** with contact stiffness, plus a −0.1 boundary-contact penalty — needed
  for swirling

Integration order per step. Note `x` uses the OLD `v`:

```
θ(t+1) = θ(t) + a_R·Δt
v(t+1) = v(t) + (a_F·h + f_drag + f_agent + f_boundary)·Δt / m
x(t+1) = x(t) + v(t)·Δt
```

`h = [cos θ, sin θ]`. `f_drag = −2·v`. `f_agent` and `f_boundary` are Hooke's law
on overlap, summed over contacts. Velocity need not align with heading after a
collision.

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| mass (both species) | 1 kg | | env edge | 2 m |
| max speed | 0.5 or 0.3 m/s | | Δt | 0.1 s |
| max linear acc | 1 m/s² | | contact stiffness | 50 N/m |
| max angular vel | 0.5 rad/s | | drag coefficient | 2 N·s/m |

**Action** `[a_F, a_R]`: `a_F ∈ [0, max_acc]` — forward only, no reverse —
and `a_R ∈ [−0.5, 0.5]` rad/s.

**Observation**: own position, velocity, heading; then relative position and
heading of the ≤6 nearest predators; then the same for the ≤6 nearest prey.
Sorted nearest-first, restricted to perception radius `R`, zero-padded to fixed
length. The topological limit of 6 comes from Ballerini 2008.

**Reward**:
```
prey     −1   while in contact with a predator
predator +1   while in contact with prey
both     −0.01·|a_F| − 0.1·|a_R|      movement cost; makes agents lazy
walls    −0.1 on boundary contact      (walls mode only)
```
**Prey are not removed when caught.** Contact is continuous "bleeding"; the
reward returns to zero on separation. Episodes are a fixed 100 steps with no
terminals.

Populations: **train** 3 predators / 10 prey. **Eval** 3 predators / 50 prey.

## Algorithm hyperparameters (Li 2023, table 2)

Networks: 3 hidden layers × 64 units, ReLU, for both actor and critic.

| | |
|---|---|
| episodes / episode length | 2000 / 100 |
| lr actor / critic | 1e-4 / 1e-3 |
| discount γ | 0.95 |
| soft update τ | 0.01 |
| replay buffer | 5e5 |
| batch size | 256 |
| initial exploration ε / noise | 0.1 / 0.1 |
| decay, per episode | `max(0.05, x − 5e-5)` |

## Metrics

`k` is the nearest conspecific to `j`. `N` is the species population.

```
DoS = (1 / (T·N·D)) · Σ_t Σ_j ‖x_j(t) − x_k(t)‖
DoA = (1 / (2·T·N))  · Σ_t Σ_j ‖h_j(t) + h_k(t)‖
```

`D` is the maximum possible separation. **On a torus of edge 2 this is √2, not
2√2** — per-axis separation caps at edge/2. Getting this wrong makes the 22%/19%
numbers incomparable.

Compute for both species even though the paper reports prey only.

**Unit test: random headings must give DoA ≈ 2/π ≈ 0.637.** The paper derives
this as `E[cos(φ/2)]`. It catches the common bug — DoA uses the *nearest
neighbour*, not the global mean heading.

Numbers to land near: DoS 22% → ~19%, DoA 0.65 → ~0.82 across training. Inside an
established flock, DoS ~15% and DoA ~0.96. Ablations: speed ratio 5:3 → DoS 18%,
DoA 0.85; 3:5 → DoS 17%; perception ⅓D → DoS 20.5%, DoA 0.75.

## Gates — the definition of done

1. **Physics** — scripted agents, no learning. Torus wrap correct, collisions
   resolve, nothing diverges, DoA unit test passes.
2. **Learning** — predator capture rate rises above the scripted baseline; prey
   capture-received falls.
3. **Emergence** — DoS falls and DoA rises across training, **and the n₀ = 0
   control stays flat.** The control is what makes this a replication rather than
   a plot.
4. **Ballpark** — DoS in 15–25%, DoA reaching 0.8+. Nice-to-have.

Gates 1–3 are the definition of done. **Exact agreement with the paper's decimals
is explicitly out of scope** — their initial-state distribution is unstated, so it
is unachievable, and chasing it burns weeks.

## Package layout

Our code is a new top-level package. `research/code_sources/` stays read-only —
PPO_example is kept pristine to diff against.

```
swarm/
├── envs/
│   ├── predator_prey.py   env + reward + PRESETS
│   ├── dynamics.py        step_motion — the extension seam
│   └── metrics.py         DoS / DoA, pure functions
├── algo/
│   ├── ddpg.py            actor, critic, replay buffer, update
│   └── config.py          every hyperparameter, with its reason
├── run/
│   ├── train.py           CLI → runs/<exp>/<preset>/s<seed>/
│   ├── eval.py            deterministic rollout of a saved policy
│   ├── plot.py            training curves
│   ├── render.py          scatter + quiver → GIF
│   └── aggregate.py       across seeds → aggregate.json, comparison.png, results/<exp>.md
└── tests/test_smoke.py    env gates + a learning gate
```

Lift from PPO_example: the normalisation wrappers, the `run/` CLI and artifact
conventions, and the smoke-test discipline. Leave `ppo/train.py` behind —
off-policy shares almost nothing with it structurally.

## Output conventions

```
runs/<exp>/<preset>/s<seed>/
├── config.json     full config + preset name — flat, so the tree is greppable
├── params.pkl      actor + critic + target nets + obs-norm stats
├── metrics.npz     per-episode arrays: return, captures, DoS, DoA, losses
├── summary.json    final numbers, wall time, steps/s, device
├── results.png     training curves
└── render.gif      one eval episode, on demand

runs/<exp>/<preset>/aggregate.json    mean ± 95% CI across seeds
runs/<exp>/comparison.png             cells overlaid
results/<exp>.md                      tracked summary table for weekly reports
```

- The experiment folder groups an ablation and maps 1:1 onto a paper figure.
  **The preset still fully determines the configuration** — values are chosen in
  `PRESETS` and `config.py` only, never in `run/`.
- **5 seeds** for every reported configuration, including the n₀ = 0 control. At
  200k steps a full cell is minutes; the gap between "it worked" and "it worked
  once" cannot be closed retroactively.
- `config.json` is duplicated inside `params.pkl` so a checkpoint stays
  self-contained. Worth the duplication — at 70 runs you will be grepping.
- **No-clobber:** `train.py` errors if the leaf exists and is non-empty, unless
  `--overwrite`. Paths stay deterministic so `aggregate.py` never has to guess
  which run is current.
- `runs/` is gitignored. `results/<exp>.md` is tracked — that is the record a
  weekly report cites.
- SLURM array index → seed, so array jobs write disjoint leaves and rsync merges
  by path.

## Phases

Ordering, not a schedule.

**0 — walking skeleton.** Package skeleton, minimal `ddpg.py`, `run/train.py`,
artifact writing, smoke test with a learning gate. Point it at **gymnax
Pendulum-v1**. Done when `runs/exp_skeleton/pendulum/s0/` holds a rising return
curve. This is a thin end-to-end slice that also happens to prove the learner
before an environment exists to blame.

**1 — the environment.** `dynamics.py`, `predator_prey.py` (torus first, Hooke
contacts, structured obs, contact reward, prey not removed), `metrics.py`,
`render.py`, and the scripted predator — turn toward nearest prey, full throttle,
the paper's own §4.4 rule. It is the gate-2 baseline, it is required for
swirling, and it exercises the env with no learning involved. Done at **gate 1**.

**2 — the replication.** Two species, parameter sharing, per-species buffers,
table 2 verbatim. `exp_flocking` and `exp_npredators` (n₀ = 0, 1, 3), 5 seeds
each. Done at **gates 2 and 3**. This is the milestone that matters.

**3 — the paper's ablations.** `exp_speed_ratio` (1:1, 5:3, 3:5),
`exp_perception` (D, ⅔D, ⅓D). `aggregate.py` and `results/*.md` land here.
Gate 4 ballpark check.

**4 — walls and swirling.** Walls boundary mode, contact stiffness, the −0.1
boundary penalty, scripted predator driving prey evolution. `exp_swirling`.

**5 — the actual goals.** Open-ended; each is now an addition, not a rewrite.
- does emergence survive massive parallelism (the deliberate version of the
  regime change deferred in phase 2)
- PPO / IPPO as a second learner
- attention encoder over the structured observation
- 3D
- drone dynamics through the `dynamics.py` seam
- marginal predation as a *measured* quantity — log distance-from-centroid of
  captured prey against the population distribution. The paper only shows this
  visually; measuring it is cheap and is a small original addition.

Phases 0–4 are the replication and the platform. Phase 5 is the internship.

## Scope of the replication

**Measurable, in scope:** fig 3 (DoS/DoA over training), fig 4 (within-episode),
fig 5b (no-predator control), fig 9 (speed ratio), fig 10 (perception range),
fig 11 (number of predators).

**Visual only, observe but do not try to measure:** fig 2 (before/after), fig 6
(confusion — the predator slows and stagnates), fig 7 (dispersion tactic, edge
effect), fig 8 (boundary aggregation vs swirling). There is no number in the
paper for "the predator looked confused."

Build the renderer early regardless. A swarm environment cannot be debugged from
scalars — a DoS of 0.31 tells you nothing about whether agents are flocking,
orbiting, or stuck in a corner. matplotlib scatter + quiver per frame → GIF.
No interactivity, no 3D, no game loop.

## Traps inherited from PPO_example

- State is a `flax.struct` pytree with **identical dtypes from `reset_env` and
  `step_env`**, or `lax.scan`'s carry fails with an opaque error.
- Static config → `pytree_node=False` → Python `if` at trace time. Traced values
  → `jnp.where` / `lax.cond`. Fixed shapes everywhere; pre-allocate and mask,
  never grow.
- **Obs-normalisation statistics are part of the policy.** They must be in
  `params.pkl`. Forgetting this looks like "trained perfectly, evaluates as
  random."
- `Environment.step` auto-resets on done and throws away the real last
  observation. Less relevant here — episodes are a fixed 100 steps with no
  terminals — but the same trap applies to anything logged at episode end.
- Smoke tests are the gate before any change is called done and before any
  cluster job.
