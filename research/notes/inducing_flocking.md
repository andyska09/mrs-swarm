# What to try next

Rewritten 2026-09-01. The previous version concluded that grouping is never
rewarded and that the replication was therefore blocked. That conclusion was drawn
before the integrator fix and through a 100-step measurement window, and it is
contradicted by the evidence in `exp_sweeps.md`. Results live there; this file is
only what to do next.

## Where the replication actually stands

Trained prey flock, significantly, against random and untrained controls. DoA
0.830 matches the paper's 0.82. Alignment survives predator removal at 80% of top
speed — the paper's C7, and comparable to Vicsek as they claim.

Three things are wrong with it:

1. **DoS does not replicate.** 0.204 against the paper's 0.19, and with predators
   removed DoS is not significantly better than random at all. We reproduce
   alignment, not cohesion.
2. **We need 5× the paper's episode length** to reach their numbers. At their
   L=100 we sit at 0.222 / 0.664, barely above random.
3. **The flock is unstable.** It forms inside 50–80 steps — as fast as the paper —
   then collapses and re-forms. The episode mean is depressed by the collapses,
   not by slow formation.

So the open question is no longer "does it flock" or "how fast does it form". It
is **why cohesion is weak and why the flock breaks.**

## Ranked, with reasons

### 1. `flocking_nocost` on the fixed physics

The one experiment that could change the interpretation of everything else, and
the cheapest. Both existing `flocking_nocost` runs predate the integrator fix.

The movement cost outweighs the survival term in every run — 77% of the reference
policy's penalty is movement, not being eaten. `cost_ar` is 0.1, ten times
`cost_af`, so *turning* is the expensive action, and the cheapest policy is to
stop turning. Agents that stop turning end up pointing the same way.

**That produces high DoA with no survival reasoning at all**, and it would explain
why alignment replicates while cohesion does not: cohesion needs thrust against
drag, which the cost punishes.

If DoA collapses without the movement cost, the flocking we are reporting is a
cost artifact and the paper's thesis is not what we are testing. If DoA holds, the
survival story survives and this doubt is closed.

One cell, ~25 minutes, config already exists and is one field off the reference.

### 2. Find out why the flock breaks

Nothing in the repo relates a collapse to anything. The instrument is missing, not
the data: `run.eval` already computes per-step DoS and DoA for every episode and
discards the array, keeping only the mean and the final-quarter mean.

Save that array. Then ask whether collapses coincide with predator approach,
with a wall (on `walls`), or happen spontaneously. "More agents stabilise it" is
already known — 3v30 and 3v50 hold DoA 0.93–0.97 in the final quarter while 3v10
scatters — which points at the group being too small to survive one scatter event.

### 3. A `updates_per_step` knob

Whether the learner is data-starved is still untested. `n_envs` was built to
answer it and cannot: it multiplies collected data while leaving the number of
gradient steps fixed, so the learner does the same amount of learning either way.
At 64 envs each transition is sampled ~0.4 times before being overwritten.

A few lines in `env_step`. Then `n_envs: 64` with `updates_per_step: 64` is the
real version of the experiment.

### 4. Train under handling time

`handling_time` froze a predator for k steps after a catch. It was the **only**
mechanism in the entire screen space that made grouping pay — capture ratio 0.97
at k=20 against 1.02–1.22 for everything else. It was removed from the code on
09-01; recover it from git history before `4c0218d1`.

The screens that found it used a prey policy trained with k=0. They measure
whether dilution helps a prey that does not know about it. **Nobody has ever
trained under `handling_time > 0`**, which is the version of the experiment that
would actually be convincing.

This is Olson's mechanism (*Predator confusion is sufficient to evolve swarming
behaviour*, J R Soc Interface 2013) and matches Li 2023 §4.3, where predators
"give up the chase, slow down and stagnate" when a prey merges into a flock. In
our environment that does not arise on its own.

If flocking improves under handling time but never under the paper's own ablation
axes, the finding is sharp: **the Li 2023 result needs a confusable predator, and
survival pressure with a competent homing predator is not sufficient.**

## Not worth repeating

The hyperparameter sweep. Seven configs, roughly eight hours, every one null:
`n_envs`, `buffer_size`, `gamma`, `expl_min`, `batch_size`, `episode_len` at 300
and at 500. None is distinguishable from the reference at a matched window.

The formation screens are also closed. `tier1` covers the paper's own ablation
space — predator count, speed ratio, prey count, arena size — across 5 spacings,
and no cell drops below 1.0. Neither does predator agility or prey speed. Only
handling time crossed.

## Screening rule

The formation test answers "does grouping pay?" in about two minutes with no
training: place prey on a grid at a chosen spacing, run a scripted predator, count
captures. A direction earns a training cell only if it drives

    capture(tight) / capture(spread)  <  1

Anything that leaves the ratio at 1.0 cannot produce flocking through the survival
reward, whatever else it changes. Use `spawn: "lattice"` with `spawn_spacing`.
