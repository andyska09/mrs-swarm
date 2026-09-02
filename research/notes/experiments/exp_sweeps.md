# Every experiment, in order

Last updated 2026-09-02. Supersedes `exp_learner_sweep.md`, which is kept for its
narrative but has three claims corrected there.

**All numbers also live in [`tables.md`](tables.md), data only.** The tables below
are duplicated from it; if they ever disagree, `tables.md` is the one regenerated
from the files.

Two kinds of experiment, and they must not be confused:

- **Training** — the learner runs, weights change, a folder appears in `runs/`.
- **Eval** — nothing learns. A frozen policy is loaded and measured. A file
  appears in `evals/`, carrying a snapshot of the config that produced it.

## The window trap — read this before any table

DoS and DoA are averaged over the steps of an episode. Prey start scattered, so
the opening of every episode is "not flocking yet". A 100-step average is mostly
that opening; a 500-step average is mostly the settled flock.

So **the same policy scores differently at different episode lengths**. The
reference policy gives DoA 0.664 measured over 100 steps and 0.698 over 300.
Nothing changed but the window.

Consequence: **training-log numbers are only comparable within one `episode_len`.**
Any cross-`L` comparison must go through `run.eval` at a matched window. Reading
the training logs across `L` is the single mistake that has cost this project the
most time.

## Training runs

3 predators vs 10 prey, torus edge 2.0, 2000 episodes, 5 seeds, MADDPG,
`actor_reg` 0.001 unless stated. Only the listed field differs from its base.

| started | config | change from base | physics |
|---|---|---|---|
| 08-29 14:59 | `flocking` | the first baseline, 3 seeds | old |
| 08-29 15:50 | `flocking` | 5 seeds | old |
| 08-29 17:49 | `flocking` | rerun | old |
| 08-30 11:45 | `flocking_reg` | `actor_reg` 0.001 | old |
| 08-30 12:42 | `flocking_nocost` | `cost_af`, `cost_ar` → 0 | old |
| 08-30 13:18 | `flocking_scripted` | fixed-rule predator, only prey learn | old |
| 08-30 14:41 | `flocking_reg` | rerun after a logging change | old |
| 08-30 14:56 | `flocking_nocost` | rerun | old |
| 08-30 15:11 | `flocking_scripted` | rerun | old |
| **08-30 19:46** | **`flocking_reg`** | **same config, rerun on fixed physics — THE REFERENCE** | new |
| 08-30 20:01 | `flocking_scripted` | rerun on fixed physics | new |
| 08-31 10:57 | `flocking_envs64` | `n_envs` 1 → 64 | new |
| 08-31 14:15 | `flocking_shortbuf` | `buffer_size` 5e5 → 8000, `n_envs` 1 | new |
| 08-31 14:29 | `flocking_gamma99` | `gamma` 0.99 **and** `n_envs` 64 — two changes | new |
| 08-31 14:57 | `flocking_explore` | `expl_min` 0.05 → 0.15, `n_envs` 64 | new |
| 08-31 15:24 | `flocking_batch1024` | `batch_size` 1024, `n_envs` 64 | new |
| 08-31 16:07 | `flocking_long300` | `episode_len` 300, `n_envs` 64 | new |
| 09-01 16:16 | `flocking_long500` | `episode_len` 500, `n_envs` 32, buffer 2e6, batch 512 | new |
| 09-01 23:06 | `flocking_long300_gamma99` | `gamma` 0.99 off `long300`, one field | new |
| 09-02 10:10 | `flocking_long300_nocost` | `cost_af`/`cost_ar` → 0 off `long300`. **1 seed** | new |

Wall time for the two long runs: `long500` 6969 s, `long300_gamma99` 6695 s, both
5 seeds on CPU.

**Trap: nothing in the metadata tells you which physics a run used.** The
integrator fix lived uncommitted in the working tree from 08-30 evening until
09-01 15:48, so every 08-30/08-31 run records commit `7b5c983a`. Worse, the two
`flocking_reg` runs at 14:41 and 19:46 share config hash `2c723019` — the config
did not change, only the code. The boundary is the timestamp, and the evidence is
that the same config gives captures 0.1017 before and 0.0747 after.

`flocking_nocost` has **only ever been run on the broken physics.** Both of its
runs predate the fix. It has never been tested since.

### Training results, last 200 episodes, ±95% CI across seeds

**Source: `runs/<dir>/s*/metrics.npz`, `[-200:].mean()` per seed, then mean ± 2.78
× SEM across the 5 seeds.** Comparable **only down a column within the same `L`.**

| run dir | config | L | DoS | DoA | cap/step | prey return |
|---|---|---|---|---|---|---|
| `20260829-155023_28ffe830` | `flocking` | 100 | 0.2211 ± 0.0066 | 0.6743 ± 0.0160 | 0.1059 | −5.73 |
| `20260830-194629_2c723019` | `flocking_reg` **(reference)** | 100 | 0.2235 ± 0.0033 | 0.6634 ± 0.0122 | 0.0747 | −3.21 |
| `20260830-200111_da6cbad6` | `flocking_scripted` | 100 | 0.2220 ± 0.0026 | 0.6615 ± 0.0073 | 0.1118 | −3.84 |
| `20260831-105755_20065180` | `flocking_envs64` | 100 | 0.2215 ± 0.0024 | 0.6621 ± 0.0147 | 0.1164 | −3.41 |
| `20260831-141526_7fe0799a` | `flocking_shortbuf` | 100 | 0.2215 ± 0.0046 | 0.6645 ± 0.0058 | 0.1219 | −3.88 |
| `20260831-142953_9fdef60b` | `flocking_gamma99` | 100 | 0.2203 ± 0.0031 | 0.6585 ± 0.0120 | 0.0842 | −2.70 |
| `20260831-145707_c7ed7ac7` | `flocking_explore` | 100 | 0.2229 ± 0.0008 | 0.6613 ± 0.0017 | 0.1558 | −4.32 |
| `20260831-152445_f9404230` | `flocking_batch1024` | 100 | 0.2196 ± 0.0017 | 0.6803 ± 0.0200 | 0.1216 | −4.06 |
| `20260831-160737_4c9eefe4` | `flocking_long300` | 300 | 0.2144 ± 0.0055 | 0.7135 ± 0.0410 | 0.1710 | −10.44 |
| `20260901-230643_16e8b80b` | `flocking_long300_gamma99` | 300 | 0.2145 ± 0.0048 | 0.7407 ± 0.0740 | 0.1488 | −9.18 |
| `20260901-161634_ad74c0b7` | `flocking_long500` | 500 | 0.2012 ± 0.0075 | 0.8265 ± 0.0744 | 0.1539 | −21.43 |

Old-physics runs are omitted from this table; their dirs are in the inventory
above and their numbers are not comparable to anything here.

Two housekeeping facts found while tracing these:
`runs/20260829-145907_28ffe830` has **no `metrics.npz`** — the run aborted, the
directory is empty of results. `20260829-155023` and `20260829-174936` are
**bit-identical**: same config hash, same seeds, and training is deterministic in
both, so the second run reproduced the first exactly.

At L=100, **every cell is the same within error.** Nothing in the hyperparameter
sweep moved DoS or DoA. The apparent gains at L=300 and L=500 are the window; see
the matched evals below, where they do not survive.

### The reward is dominated by movement, not survival

Prey reward splits into a survival term (±1 per contact) and a movement cost.
**Source: the `prey_survival` and `prey_movement` keys in the same
`metrics.npz`,** last 200 episodes:

| config | survival | movement | movement's share |
|---|---|---|---|
| `flocking_reg` | −0.75 | −2.46 | **77%** |
| `flocking_long300` | −5.13 | −5.31 | 51% |
| `flocking_long500` | −7.69 | −13.74 | **64%** |

**Being eaten costs the prey less than moving does, in every run.** A prey that
holds still is behaving correctly under this reward. This sits awkwardly with the
paper's thesis that *survival* pressure drives swarming, and it has never been
tested on the fixed physics — that is what `flocking_nocost` is for.

## Eval screens

### Formation test — does being in a group help you survive?

Prey placed on a jittered grid at a chosen spacing, scripted predator, prey policy
frozen from the reference run, 100 steps, no learning. Report is the capture rate
at the tightest spacing divided by the rate at the loosest.

**Below 1.0 means grouping pays. Above 1.0 means grouping is punished.**

**Source: `evals/<screen>_*.json`, field `captures_per_step.mean`, averaged over the
`_p0..p4` prey seeds at each spacing, then tightest ÷ loosest.** Prey policy in all
of them is `runs/20260830-194629_2c723019` (the reference); predator is
`envs/scripted.py`, not a network. Spacing is the `_s015`…`_s045` suffix =
`spawn_spacing`.

| screen | what varied | best cell | ratio |
|---|---|---|---|
| `tier1` | predator count, speed ratio, prey count, arena size — 8 cells × 5 spacings | `small` (edge 1.4) | 1.02 |
| `tier1b` | the 4 surviving cells × 5 prey seeds | `npred1` | 1.02 |
| `agility` | predator turn ×1/2/4/8, predator acceleration | `turn1` | 1.03 |
| `preyspeed` | prey speed ×1, ×1.5, ×2, and the paper's 3:5 | `base` | 1.03 |
| **`handling`** | **predator frozen 0/3/5/10/20 steps after a catch** | **k=20** | **0.97** |

`tier1` covers the paper's own ablation space (§4.5, §4.7). Every cell says
grouping is neutral or harmful.

**Handling time is the only mechanism ever found that made grouping pay** — 0.99
at k=10, 0.97 at k=20. It was removed from the code on 09-01 at the user's
request; recover it from git history before `4c0218d1`. The 120 result files in
`evals/handling_*.json` cannot be regenerated, and they still describe the removed
field.

Caveat on all of these: the prey policy was trained with **k=0** and full movement
cost. The screens measure whether grouping helps a prey that never learned to
exploit it. Nobody has trained under `handling_time > 0`.

### Matched evals — the correct instrument

Uniform spawn, greedy `mu(o)`, no exploration noise, 200 episodes, 5 seeds, every
cell at the stated window. `speed` is mean prey speed; the cap is 0.5.

**Source: `evals/<cell>_s0..s4.json`, the `.mean` of each named field.** Which run
each cell loads:

| cell prefix | pred policy | prey policy |
|---|---|---|
| `matched_base`, `matched_l100base` | `20260830-194629_2c723019` | same |
| `matched_preyrandom` / `preyuntrained` / `l100rand` | `20260830-194629_2c723019` | random / untrained |
| `matched_envs64` | `20260831-105755_20065180` | same |
| `matched_long300`, `m300_l300b` | `20260831-160737_4c9eefe4` | same |
| `m300_g99` | `20260901-230643_16e8b80b` | same |
| `m500_long300` | `20260831-160737_4c9eefe4` | same |
| `m500_long500`, `nopred_long500` | `20260901-161634_ad74c0b7` | same |
| `m500_rand`, `nopred_rand` | `20260901-161634_ad74c0b7` | random |
| `nopred_long300` | `20260831-160737_4c9eefe4` | same |

`matched_long300` and `m300_l300b` are the same policy at the same window; the
rerun exists only because `prey_speed` was added to `run.eval` afterwards.

| cell | L | DoS | DoA | DoS ¼ | DoA ¼ | speed | cap/step |
|---|---|---|---|---|---|---|---|
| `matched_l100rand` | 100 | 0.2263 | 0.6392 | 0.2225 | 0.6406 | — | 0.2505 |
| `matched_l100base` | 100 | 0.2218 | 0.6641 | 0.2134 | 0.6818 | — | 0.0650 |
| `matched_preyrandom` | 300 | 0.2235 | 0.6424 | 0.2220 | 0.6414 | — | 0.2993 |
| `matched_preyuntrained` | 300 | 0.2244 | 0.6451 | 0.2225 | 0.6510 | — | 0.2940 |
| `matched_base` | 300 | 0.2129 | 0.6978 | 0.2071 | 0.7274 | — | 0.0745 |
| `matched_envs64` | 300 | 0.2147 | 0.6924 | 0.2077 | 0.7203 | — | 0.2340 |
| `matched_long300` | 300 | 0.2144 | 0.7294 | 0.2056 | 0.7796 | — | 0.2134 |
| `m300_l300b` (`long300` rerun) | 300 | 0.2144 | 0.7294 | 0.2056 | 0.7796 | 0.0840 | 0.2134 |
| `m300_g99` | 300 | 0.2126 | 0.7814 | 0.2020 | 0.8564 | 0.0766 | 0.1597 |
| `m500_rand` | 500 | 0.2239 | 0.6404 | 0.2236 | 0.6394 | 0.2437 | 0.6701 |
| `m500_long300` | 500 | 0.2086 | 0.7555 | 0.1987 | 0.7963 | 0.0856 | 0.2862 |
| `m500_long500` | 500 | 0.2035 | 0.8304 | 0.1925 | 0.8597 | 0.0853 | 0.1775 |

Paired over seeds, df = 4, significant at |t| > 2.78:

| comparison | DoS ¼ | DoA ¼ |
|---|---|---|
| `long300` − reference, at L=300 | −0.0015, t = −0.31 | +0.0521, t = +0.94 |
| `long500` − `long300`, at L=500 | −0.0062, t = −0.82 | +0.0634, t = +1.01 |
| `long500` − random, at L=500 | −0.0311, **t = −8.92** | +0.2203, **t = +4.82** |
| `long300` − random, at L=500 | −0.0249, **t = −3.91** | +0.1569, **t = +3.40** |
| `gamma99` − `long300`, at L=300 | −0.0036, t = −0.57 | +0.0768, t = +1.23 |

**Trained prey flock. No training setting flocks better than another.** Every
trained policy is far from random on both metrics; no config change is
distinguishable from the reference on DoS or DoA. Seven sweep runs, roughly ten
hours of compute, all null on flocking.

**Removing the movement cost is the first thing that has moved flocking.** Seed 0
only, so this is a lead and not a result. `flocking_long300_nocost` against
`flocking_long300` on the same seed: DoA 0.824 vs 0.677, DoS 0.200 vs 0.219,
captures 0.133 vs 0.165. The +0.147 in DoA is larger than any effect in the whole
sweep and exceeds all five of `long300`'s seeds (max 0.767). Prey thrust 2.3× and
turn 2.5× harder, the predator goes from 40% to 70% throttle — and captures still
fall. DoA was **still climbing at episode 2000** (0.722 → 0.778 → 0.824 over the
last 600), so 2000 episodes may not be enough. In the L=500 replay it forms slower
(DoA > 0.9 at step 153 against 79) but does **not collapse**, holding 0.908 in the
final quarter where `long300` falls from 0.979 to 0.665.

Two things changed at once: the prey were freed *and* the predator was unleashed.
`flocking_freeprey` (`prey_cost_scale: 0.0`, predator still pays) separates them
and is unrun. Four more seeds of `nocost` are the other missing piece.

**`gamma` 0.99 is the one setting with a real effect — on survival, not flocking.**
`flocking_long300_gamma99` is one field off `flocking_long300`, and at a matched
L=300 window it gives captures −0.0537 (t = −3.00) and prey return +2.11
(t = +3.02), both significant, while DoS and DoA stay null. The same pattern
appeared at L=100: `flocking_gamma99` cut captures 28% and moved DoS/DoA not at
all. Longer horizon buys better evasion and no cohesion.

Caveat: both species come from the same run, so "captures fell" cannot be split
into "prey evade better" versus "predator hunts worse". Cross-matching the two
runs' species would settle it and has not been done.

Untrained prey (`matched_preyuntrained`) behave like random prey — the effect is
learned, not an artifact of network initialisation.

### Predator removal — the paper's C7 control

**Source: `evals/nopred_{rand,long300,long500}_s0..s4.json`; policies as in the
table above.** Same policies, `n_pred: 0`, L=500. Tests whether alignment is a
flock or merely a reaction to being chased. The paper (§4.2, fig 5a) reports prey keep high DoA with
predators removed, "comparable to the well-known Vicsek model".

| cell | DoS ¼ | DoA ¼ | speed |
|---|---|---|---|
| `nopred_rand` | 0.2258 | 0.6312 | 0.2445 |
| `nopred_long300` | 0.1881 | 0.8533 | 0.4092 |
| `nopred_long500` | 0.2081 | 0.7845 | 0.3920 |

Against random, paired: DoA +0.222 (t = +4.75) for `long300`, +0.153 (t = +4.14)
for `long500`. Both significant. **DoS is not significant in either** (t = −2.34,
−1.82).

**C7 replicates, and it replicates as alignment only.** With no predators the prey
hold heading together at 80% of top speed, but do not clump more than random. That
is precisely a Vicsek result, which is what the paper compares it to.

Also: **removing predators makes prey 4.6× faster** — 0.085 while hunted, 0.39–0.41
when free. Hunted prey barely move. That follows from the reward split above.

## Against the paper

**Source: paper column from `li2023_spec.md` claims C2–C5 (§4.2, figs 3–4); our
columns are `matched_l100base` / `matched_l100rand` and `m500_long500` / `m500_rand`
from the matched-eval table above.** Paper targets are all at their L=100.

| | paper | us, L=100 | us, L=500 |
|---|---|---|---|
| DoS, start / random | 0.22 | 0.226 | 0.224 |
| DoA, start / random | 0.65 | 0.639 | 0.640 |
| DoS, trained | **0.19** | 0.222 | 0.204 |
| DoA, trained | **0.82** | 0.664 | **0.830** |

The starting point matches exactly, so the environment and metrics are right.
**DoA is replicated. DoS is not.** And we need 5× the paper's episode length to
reach their DoA at all.

## Within-episode behaviour — the flock is fast but unstable

**Source: `eval_configs/curve_long300.json` and `curve_long500.json` → `run.replay`
→ `renders/curve_*/traj.npz`, then `metrics.dos`/`doa` per step.** Single episode,
3v10, L=500, env seed 0, each run's best seed (`long300` seed 3, `long500` seed 2).

| step | 0 | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 499 |
|---|---|---|---|---|---|---|---|---|---|---|
| `long300` DoS | 0.201 | 0.246 | 0.209 | 0.148 | 0.264 | 0.207 | 0.229 | 0.158 | 0.172 | 0.133 |
| `long300` DoA | 0.525 | 0.642 | 0.826 | 0.879 | 0.899 | 0.955 | 0.971 | 0.976 | 0.979 | 0.665 |
| `long500` DoS | 0.202 | 0.159 | 0.136 | 0.132 | 0.115 | 0.138 | 0.168 | 0.219 | 0.123 | 0.219 |
| `long500` DoA | 0.514 | 0.670 | 0.934 | 0.956 | 0.980 | 0.985 | 0.943 | 0.843 | 0.985 | 0.961 |

DoA first crosses 0.9 at **step 79** (`long300`) and **step 43** (`long500`).
`long500` reaches DoS 0.136 by step 50 and 0.115 by step 100; the paper's fig 4
reaches 0.15 by about step 30.

**Correction to `exp_learner_sweep.md`: formation is not 5–10× slower than the
paper's.** That claim came from averaged curves, where episodes forming at
different moments smear a sharp rise into a slow ramp. In a single episode the
flock forms within 50–80 steps, comparable to the paper.

**The real defect is instability.** `long300` holds DoA 0.98 through step 400 then
collapses to 0.665 by 499. `long500` peaks at 0.985, drops to 0.843 by step 250,
recovers to 0.985 at 400. The episode mean of 0.87–0.91 against peaks of 0.99 is
measuring how often the flock falls apart, not how long it takes to form.

## What we know

1. The integrator fix was the one intervention that mattered. Forward Euler made
   collisions create energy — an anti-clustering force unrelated to predation.
2. Trained prey flock, significantly, against random and untrained controls.
3. The flocking is alignment, not cohesion. DoA replicates the paper; DoS does not,
   and without predators DoS is not significant at all.
4. Alignment survives predator removal at speed — the paper's C7.
5. No hyperparameter changed flocking. Seven sweeps, all null on DoS and DoA. The
   only setting with a measured effect is `gamma` 0.99, and it buys survival
   (captures −25%, prey return +2.11 at a matched window, both significant) while
   leaving DoS and DoA untouched. Evasion and cohesion are separable, and every
   knob we have moves only the first.
6. The flock forms fast and then breaks. Stability is the open defect.
7. The movement cost outweighs the survival term in every run.
8. In the entire screen space, only handling time made grouping pay — and it is
   now removed from the code.

## What we do not know

- Whether the flocking is driven by survival at all, or is a side effect of the
  turning cost. **Partly answered on 09-02**: with the cost removed, DoA rose from
  0.677 to 0.824 on the same seed, so the cost was *suppressing* flocking rather
  than manufacturing it. One seed. Needs four more, plus `flocking_freeprey` to
  say whether the prey's cost or the predator's was doing it.
- Why the flock breaks. Nothing correlates collapse events with predator attacks.
- Whether prey trained under `handling_time > 0` would exploit dilution.
- Whether the learner is data-starved. `n_envs` was meant to test this and cannot:
  it multiplies data without changing the number of gradient steps, so the learner
  does exactly as much learning either way. See the replay-ratio note below.

## Traps

- **`n_envs` does not increase learning.** Gradient steps are unchanged; only rows
  collected go up. At 64 envs each transition is sampled ~0.4 times before being
  overwritten, against ~25.6 at 1 env — most collected data is never used. Testing
  data-starvation needs a `updates_per_step` knob that does not exist.
- **`flocking_gamma99` is not a clean experiment.** It changes `gamma` *and*
  `n_envs`. `flocking_long300_gamma99` is the clean one.
- **`buffer_size` counts rows, not steps.** At 64 envs a 5e5 buffer spans ~8
  episodes instead of ~500.
- Training metrics are means over the episode. The within-episode curve is computed
  by `run.eval` and then discarded; only the mean and final-quarter mean survive.
- `evals/*.json` embeds the config that produced it. That is why the generated
  sweep config folders could be deleted, and why results survived the
  `speed_pred` → `max_speed_pred` rename and the `handling_time` removal.
