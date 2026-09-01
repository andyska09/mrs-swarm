# How to induce flocking — directions, ranked

Written after the 2026-08-30 cells. The companion file
`replication_gap_suspects.md` says why the baseline fails; this one says what to
try next. Everything here is grounded in a measurement, not a guess.

## The diagnosis in one paragraph

Flocking does not emerge because **grouping buys the prey nothing**. Formation
test (prey placed in a disc of radius `rc`, 200 episodes per condition, no
training): capture rate at `rc = 0.15` divided by capture rate at `rc = 1.0` is
**1.06 ± 0.02** against the learned predator and **1.08–1.30** against the
scripted one. Tight is never better than spread. With no survival gradient along
the density axis, a reward-maximising prey has no reason to aggregate, and ours
correctly does not. The learner works (`actor_reg` fix), the predator is a real
hunter (0.232–0.292 captures/step vs 0.136 random), and the prey learn strong
evasion (2.6× better than any naive policy). All of that is measured. The one
missing thing is the payoff.

## What the literature says the missing ingredient is

Olson et al. name it directly, and it is not survival pressure on its own:

- **Predator confusion is sufficient to evolve swarming behaviour** (Olson,
  Hintze, Dyer, Knoester, Adami, *J R Soc Interface* 2013, arXiv:1209.3330).
  Prey hunted by a *confusable* predator evolve dense swarms; prey hunted by a
  non-confusable one do not. Confusion is the selection pressure, not predation
  as such.
- **Evolution of swarming behavior is shaped by how predators attack**
  (*Artificial Life* 22(3) 2016, arXiv:1310.6012): "how predators attack is
  critical to the evolution of the selfish herd", and **density-dependent
  predation** — predator success falling as prey concentration rises — is what
  gives the selfish herd its advantage.

Our environment has the opposite sign. A cluster is a *persistent contact zone*:
a homing predator inside a group touches someone almost every step, while among
spread prey it spends most of its time travelling with no contact at all. That
is radius-independent — swept 6× (contact distance 0.150 → 0.025 m), the ratio
never drops below 1 and mostly grows.

Li 2023 does claim the mechanism exists in its own runs (§4.3: predators "give up
the chase, slow down and stagnate" when a prey merges into a flock; dispersion
tactic; marginal predation). In ours it does not arise. §4.2 also hedges the
causal story — flocking is "largely an outcome of passive space extrusion and
polarization induced by predators" — i.e. even the paper does not claim grouping
pays directly.

## One encouraging measurement

Within a single episode the right trend *is* present, about 5× too weak:

| step | 0 | 25 | 50 | 75 | 99 |
|---|---|---|---|---|---|
| DoS | 0.237 | 0.231 | 0.223 | 0.223 | 0.222 |
| DoA | 0.617 | 0.643 | 0.654 | 0.675 | 0.682 |

Prey do drift together and align slightly over 10 s, and DoA is still rising at
the end. Target is DoS 0.19 / DoA 0.82.

## The screening instrument

**Do not train first.** The formation test answers "does grouping pay?" in ~2
minutes with no learning at all, and every direction below can be screened with
it. A direction is worth a training cell only if it drives

    capture(tight) / capture(spread)  <  1

Anything that leaves the ratio at 1.0 cannot produce flocking through the
survival reward, whatever else it changes.

## Tier 1 — screen with the formation test, no training

| # | Change | Mechanism it would create | Config |
|---|---|---|---|
| 1 | Predator count `n_pred` 1 / 3 / 5 / 8 | More hunters may make a cluster genuinely defensible (or worse). §4.7 says more predators speed emergence | `n_pred` |
| 2 | Speed ratio 5:3 and 3:5 | §4.5 reports 3:5 (prey faster) gives the *lowest* DoS, 17% — prey "evade while maintaining formation" | `speed_pred` / `speed_prey` |
| 3 | Prey count 10 → 50 | The paper's own evaluation density. Neighbours are 0.14 m away at random instead of 0.32 m, so a flock is reachable inside one episode | `n_prey` |
| 4 | Arena edge 2.0 → 1.4 | Same density effect without changing population | `edge` |

## Tier 2 — training cells, only for whatever survives tier 1

| # | Change | Why | Cost |
|---|---|---|---|
| 5 | `episode_len` 100 → 300–500 | DoA is still climbing at step 99. Aggregation is slower than the episode, and the episode average is diluted by the random start | 1 cell, ~45 min |
| 6 | `gamma` 0.95 → 0.99 | Horizon is 20 steps = 2 s. Closing a 0.2 m gap takes ~10 steps and only pays off later. A benefit spread over 10 s is invisible at γ = 0.95 | 1 cell |
| 7 | Paper's §4.5 / §4.7 ablations as training cells | Reproduces the paper's own strongest-swarming conditions | 2–3 cells |

## Tier 3 — inject the missing mechanism deliberately

If tier 1 finds no regime where grouping pays, then under this dynamics model
survival pressure alone is *not* sufficient, and that is the finding. Test it
by putting the mechanism in by hand — each of these is a few lines in
`envs/`:

| # | Mechanism | Implementation |
|---|---|---|
| 8 | **Handling time** | After a contact, the predator is frozen for `k` steps. Creates classic dilution: your neighbour being caught protects you |
| 9 | **Confusion** | Corrupt the predator's target choice in proportion to the number of prey within some radius — noise on the observed prey positions, or a random target swap. This is Olson's mechanism |
| 10 | **Density-dependent capture** | Capture succeeds with probability falling in local prey count. The bluntest version of the same thing |

8 and 9 keep the reward swarm-independent, so the paper's thesis survives; what
changes is the *predator*, which is where Olson puts it. If flocking appears
under 8 or 9 but never under tiers 1–2, the thesis contribution is sharp: **the
Li 2023 result requires a confusable predator, and survival pressure with a
competent homing predator is not sufficient.**

## Measurement changes to make regardless

- Log DoS/DoA over the **final quarter** of the episode as well as the whole
  episode. Eq (2) averages from a random start, which costs ~0.008 of DoS.
- Log the within-episode DoS/DoA curve for the final policy, not just the
  episode mean. The trend above is invisible in the episode average.

---

## Tier 1 result, 2026-08-30: every condition fails

Screen run as designed — prey on a jittered lattice of spacing `s`, scripted
predator, 100 episodes per cell, reporting the fraction of prey in contact per
step. `s = 0.10` is a tight flock (just above the 0.08 m contact distance),
`s = 0.45` is spread over most of the arena.

**Stationary prey** (pure geometry, no policy confound):

| condition | s=0.10 | 0.15 | 0.22 | 0.32 | 0.45 | tight/spread |
|---|---|---|---|---|---|---|
| base 3v10, 1:1 | 0.0398 | 0.0346 | 0.0312 | 0.0280 | 0.0272 | 1.47 |
| n_pred = 1 | 0.0171 | 0.0131 | 0.0116 | 0.0097 | 0.0088 | 1.94 |
| n_pred = 5 | 0.0608 | 0.0591 | 0.0531 | 0.0467 | 0.0433 | 1.41 |
| n_pred = 8 | 0.0888 | 0.0838 | 0.0788 | 0.0757 | 0.0688 | **1.29** |
| speed 5:3 | 0.0455 | 0.0388 | 0.0339 | 0.0293 | 0.0266 | 1.71 |
| speed 3:5 | 0.0389 | 0.0368 | 0.0349 | 0.0329 | 0.0326 | 1.19 |
| n_prey = 50 | 0.0250 | 0.0239 | 0.0216 | 0.0207 | 0.0209 | 1.19 |
| edge = 1.4 | 0.0553 | 0.0535 | 0.0510 | 0.0458 | 0.0432 | 1.28 |

**Trained prey** (policy from the base cell, so off-distribution everywhere else):
base 1.15, n_pred=1 1.28, n_pred=5 1.13, n_pred=8 1.10, 5:3 1.16, 3:5 1.38,
n_prey=50 1.16, edge=1.4 1.11.

**Not one condition drops below 1.0.** The ratio is monotone in every cell:
tighter is always worse. More predators move it toward 1 (1.94 → 1.29) and so
does higher density, but nothing crosses.

That closes tier 1 and, with it, the paper's own ablation space. `n_pred`,
the speed ratio, the population and the arena size are the four axes Li 2023
varies in §4.5 and §4.7, and none of them creates a survival advantage for
grouping in this dynamics model.

**Therefore: skip tier 2.** Longer episodes and a longer discount horizon only
help the learner *find* an advantage that exists. There is no advantage to find.

Go to tier 3. Handling time (#8) is the cheapest and the most classic — freeze a
predator for `k` steps after a contact and the selfish herd becomes real, because
a neighbour being caught genuinely protects you. Confusion (#9) is the one
Olson's result and Li 2023 §4.3 both point at, and is the better match to the
paper's own description of what its predators do.
