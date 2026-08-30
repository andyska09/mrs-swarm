# Replication gap — why swarming does not emerge

## Status, 2026-08-30

Suspect 1 is **closed**: diagnosed, fixed, verified. Suspect 2 is now the live one
and is finally testable, because a working learner is a precondition for reading
any spatial metric at all.

Everything below the divider is the original static audit of `swarm/`. It is kept
as written; where it is now superseded the suspect header says so.

### What happened

`swarm_simple/` reproduces the pathology exactly, on code that shares nothing with
`swarm/`. So it was never a bug in the old implementation — it follows from the
spec'd environment, reward and MADDPG update.

`maddpg.py` now logs `prey_af` / `prey_ar` per episode, so the old back-calculation
of |a_R| from the movement cost is a direct measurement. It agreed to three
decimals: predicted 0.447, measured 0.447 (`runs/20260829-174936_28ffe830`,
flocking, 5 seeds).

That run also killed the "laziness" reading of suspect 2. The two action channels
move in opposite directions: prey drive `a_F` down (0.290 → 0.217, cost 0.01/unit)
and `a_R` up to the cap (0.266 → 0.447, cost 0.1/unit). An agent finding the lazy
optimum kills the expensive channel first. These prey do the reverse, so the
saturation cannot be a response to the reward scale.

### The mechanism, measured

Greedy rollouts from the checkpoints, pre-tanh logits recovered by `arctanh`:

| prey, `a_R` channel | baseline | with fix |
|---|---|---|
| mean \|z\| | 5.49 | 0.61 |
| frac \|a_R\| > 0.99 | 0.76 | 0.01 |
| tanh slope, the surviving gradient | 0.094 | 0.715 |

|z| of 5.5 puts the actor where `1 − tanh²(z)` is ~0; for most samples it is zero
to float32 precision. The DDPG actor objective `−Q(o, μ(o))` contains no term in
`z`, so nothing opposes the drift, and Adam divides out the vanishing gradient so
the drift does not slow. The critic cannot supply the missing correction either:
the movement cost is worth |dQ/da_R| ≈ 0.05, against a critic RMSE of
√0.015 = 0.12. Advice 2.4× smaller than the noise around it is a random walk, and
a random walk with no restoring force reaches an absorbing boundary.

### The fix

Two learner-only changes, env untouched, both switchable from config
(`actor_reg`, `target_noise`, `target_noise_clip`, all `0.0` in `flocking.json`):

- `actor_reg = 1e-3` on `mean(z²)` in the actor loss. Reference MADDPG
  (`openai/maddpg`, `p_train`) carries this term at the same weight — verified in
  source. Caveat: their actor has no tanh (Gumbel-softmax over discrete MPE
  actions), so the term is structurally the same but not aimed at this pathology.
  The justification here is our own measurement.
- `target_noise = 0.2`, clipped at `0.5`, on the target action (TD3 smoothing).

### Result: the learner works, the swarming still does not

`runs/20260830-114547_9ebf0e65` (flocking_reg, 5 seeds), final 100 episodes
against the baseline:

| per prey per episode | baseline | fixed |
|---|---|---|
| `prey_reward` | −5.77 | **−3.39** |
| `prey_movement` | −4.68 | **−2.46** |
| `prey_survival` | −1.08 | −0.93 |
| `prey_ar` (physical) | 0.447 | **0.219** |
| `captures` | 0.108 | 0.093 |
| `dos` | 0.222 | **0.223** (random: 0.226) |
| `doa` | 0.680 | 0.674 (random: 0.637) |

`prey_z` never climbs — flat at 0.86 from episode 100 on. Both species now improve
monotonically on their own objective; predators go −1.67 → −0.36 with `pred_q`
staying positive. And DoS does not move one thousandth off the random value.

This is the useful part. A frozen actor explains a null result trivially; a
working actor that optimises for 2000 episodes and still produces a uniformly
random configuration says something real — **under this reward, at these radii,
not flocking is correct behaviour.** Prey still spend 2.46 per episode on movement
against 0.93 of survival, so evasion continues to cost more than being caught.

Both fixes shipped in one cell, so attribution between them is untested. `learn`
now splits its key, so `flocking.json` no longer reproduces
`20260829-174936_28ffe830` bitwise; it is still a valid control.

### Next

Suspect 2, directly: `cost_af = cost_ar = 0` (survival pressure alone), then
`catch_reward = 5.0`. If DoS moves off 0.226 in either, the reward balance is the
binding constraint and the free parameter in open question 1 is what decides the
whole replication.

---

Forensic audit of `swarm/` against Li, Li & Zhao 2023 (New J. Phys. 25 092001),
transcript `research/papers/Li_2023_New_J._Phys._25_092001.md`. Static reading
only — nothing was run. Numbers quoted from `results/*.md` and from the
`summary.json` files already on disk under `runs/`.

## TL;DR

Every DoS number in `results/*.md` is, to three decimals, the closed-form
expectation for a **uniformly random** configuration in that arena — 0.226 for
10 prey on the 2 m torus, 0.131 for 10 prey in the walled box (which is the same
configuration, just divided by 2√2 instead of √2), 0.426 for 3 predators, and
0.100 for the 50-prey evaluation. DoA sits at 2/π = 0.637 plus a few percent
everywhere. So nothing in the whole experiment grid ever produced non-random
spatial structure; the metrics are not lying, there is simply nothing to see.

The cause is two-layered. **The learner is diverging, not learning:** prey
episodic reward goes −3.45 → −5.88 and predator reward −0.12 → −1.96 over 2000
episodes, captures/step stays flat at 0.101 → 0.105, and the prey actor drives
|a_R| from ~0.22 to ~0.45 rad s⁻¹ — saturating the single most expensive term in
its own reward. **And the reward scale makes evasion pointless anyway:** the
movement cost is −4.83 per prey-episode against a survival term of −1.05, so a
prey that simply stops moving beats every evasive policy by a factor of four.
The one cell that shows genuine cohesion (`swirl`, greedy eval: nearest-neighbour
distance 0.257 m vs 0.370 m random, DoA 0.805) is the only cell whose predator is
the paper's scripted rule instead of a learned DDPG agent. Run
`scripted_predator=True` on the **torus** and you will know within an hour which
half is broken.

Aggravating factor: commit `8acc494` deleted `swarm/algo/train_gymnax.py` and
`ce84686` deleted the Pendulum `learning_gate()` that asserted
`late > early + 300` — the only test that ever checked the DDPG update *improves
a policy*. Its replacement (`swarm/tests/test_smoke.py:220`) asserts
`pred_q > 0 > prey_q` on an 80-episode run; the real 2000-episode run finishes
with `pred_q_final = -0.048`, i.e. the gate passes only because it stops early.

## The decisive arithmetic

`swarm/envs/metrics.py:22-35` implements eq (2) correctly, so DoS is a faithful
estimator of mean nearest-neighbour distance / D. For N points uniform on a
torus of edge L, P(nearest of the other N−1 beyond r) = (1 − πr²/L²)^(N−1), so
E[d] = ∫₀^∞ (1 − πr²/L²)^(N−1) dr. Integrating that (and cross-checking by Monte
Carlo, including hard-core exclusion at the preset radii):

| condition | analytic / MC **random** DoS | observed "first" | observed "final" |
|---|---|---|---|
| torus, 10 prey, r=0.04 | **0.2334** (hard-core) / 0.2264 (points) | 0.227 | 0.217 |
| torus, 10 prey, r=0.06 (`rad15`) | **0.2395** | 0.231 | 0.223 |
| torus, 10 prey, r=0.08 (`rad20`) | **0.2481** | 0.239 | 0.230 |
| torus, 10 prey, r=0.12 (`rad30`) | **0.2703** | 0.257 | 0.247 |
| torus, 3 predators | **0.4258** | 0.4156 (`pred_dos_first`) | 0.4182 |
| walls, 10 prey, D = 2√2 | **0.1308** | 0.131 | 0.123 |
| torus, 50 prey (`eval_eval50.json`) | **0.1004** | — | 0.100 |
| DoA, uniform headings | **2/π = 0.6366** | 0.626–0.659 | 0.635–0.692 |

Read the radius column top to bottom: the entire `exp_radius` "effect"
(0.227 → 0.257) is reproduced to ±0.013 by nothing but excluded volume of larger
disks. It is geometry, exactly as suspected. The `swirl` row is the same
configuration as the torus row, renormalised: `metrics.py:34` divides by
`edge*√2/1 = 2√2` when `periodic=False` and by `edge*√2/2 = √2` when it is true,
so the walls cells look 2× better and are not comparable to anything.

And the `eval50` number is a reporting trap in the other direction: DoS = 0.100
with 50 prey looks far better than the paper's 19%, but 0.1004 *is* the random
value at N = 50. DoS is not density-normalised (`metrics.py:28-29` says so), so
the paper's 22% → 19% → 15% figures are all at n₁ = 10; figure 4's 15% cannot be
an n₁ = 50 number because 0.15 > 0.100 would mean *more dispersed than random*.

## Prime suspects, ranked

### 1. Neither species improves on its own reward — the learner diverges

> **CLOSED 2026-08-30.** Confirmed on the facts and on the mechanism; the guessed
> mechanism ("critic noise driving a random walk into tanh saturation") was right.
> Fixed by `actor_reg` + target smoothing. See the status section at the top. The
> text below is the original static reading and is superseded.

*Claim.* Over 2000 episodes both actors get monotonically **worse** under their
own objective, and the prey actor saturates the highest-penalty action channel;
so the policies never leave the neighbourhood of random behaviour and DoS stays
at the random-configuration value.

*Evidence in the code / on disk.* `runs/exp_flocking/flocking/s0/summary.json`:

```
prey_reward_first   -3.453   ->  prey_reward_final   -5.880
prey_survival_first -1.012   ->  prey_survival_final -1.050
prey_movement_first -2.441   ->  prey_movement_final -4.830
pred_reward_first   -0.117   ->  pred_reward_final   -1.961
pred_q_first        +1.565   ->  pred_q_final        -0.048
captures_first       0.101   ->  captures_final       0.105
```

`runs/exp_duration/long/s0/summary.json` (2 M steps) adds the action channels:

```
prey_af_first 0.240 -> prey_af_final 0.359       (a_F, max 1.0)
prey_ar_first 0.220 -> prey_ar_final 0.471       (|a_R|, max 0.5)
pred_ar_first 0.317 -> pred_ar_final 0.456
```

`prey_ar` is `jnp.abs(action_phys[:,1])` (`swarm/envs/predator_prey.py:197`), the
*physical* rad s⁻¹, capped at `max_ang_vel = 0.5` (`predator_prey.py:51, 217`).
0.471 is 94% of the cap. The reward charges `cost_ar = 0.1` per unit of it
(`predator_prey.py:186-187`):

```python
    movement = -(params.cost_af * jnp.abs(action_phys[:, 0])
                 + params.cost_ar * jnp.abs(action_phys[:, 1]))
```

The same arithmetic recovers the flocking run, which predates the `a_r` logging:
`prey_movement_final = -4.83` over 100 steps ⇒ 0.01·a_F + 0.1·|a_R| = 0.0483 ⇒
|a_R| ≈ 0.447. Contrast `runs/exp_npredators/npred0/s0/summary.json`, where the
contact term is identically zero:

```
prey_movement_first -0.811 -> prey_movement_final -0.352   =>  |a_R| <= 0.035
prey_q_final -0.0042 ,  prey_critic_loss_final 3.0e-09
```

**Without predators the actor correctly finds the lazy policy. With predators it
saturates the turn rate.** That is the whole diagnosis in two runs.

The predator side is worse: at `pred_q_first = +1.565` the critic values a policy
whose true discounted value is `pred_reward_first/100/(1-γ) = -0.023`. A 68×
overestimate on a sparse +1 reward is the textbook DDPG bootstrap blow-up, and
`swarm/algo/ddpg.py:152-179` has no twin critic, no target-policy smoothing and
no gradient clipping to stop the actor climbing it.

*What the paper says.* §3.4: "prey receive a reward of −1 if caught by predators
while predators receive a reward of +1 … A decorative reward that mimics energy
consumption due to movement is added … −0.01|a_F| − 0.1|a_R|. This reward
function will cause the agent to exhibit laziness." §4.3 and figures 6–7 report
that predators become *better* pursuers (dispersion tactic, marginal predation);
figure 3 shows a monotone improvement over 2000 episodes.

*Why it produces the observed numbers.* An agent at |a_R| = 0.45 rad s⁻¹ with
a_F = 0.36 has drag-limited speed a_F/drag = 0.18 m s⁻¹
(`swarm/envs/dynamics.py:27-30`, drag = 2), so it traces a circle of radius
v/ω = 0.18/0.45 = **0.40 m** and closes ~75% of it in a 100-step episode. Every
prey orbits its own spawn point. A rotation of a uniform distribution is still
uniform, so the ensemble DoS stays pinned at 0.2264 forever, and neighbours'
orbital phases are uncorrelated, so DoA stays at 2/π. This is the direct
mechanical link between the actor pathology and the 0.227 → 0.217 plateau.

*Confidence.* **High** on the facts, **medium** on the mechanism (critic noise
from the sparse ±1 term swamping the −0.05·sign(a_R) movement gradient, driving a
random walk into tanh saturation). Cheapest kill: open
`runs/exp_duration/long/s0/results.png`, already on disk — if `prey_ar` rises
monotonically to the cap while `prey_reward` falls, it is confirmed with no
compute at all. Second cheapest: restore the deleted Pendulum learning gate
(`git show a55ae17:swarm/algo/train_gymnax.py`) and confirm `ddpg.py` still
raises return on a dense single-agent task.

---

### 2. The movement cost outweighs the survival signal ~5:1, so evasion is never worth its price

> **LIVE, and now the prime suspect.** Two updates. The "laziness" reading is dead
> — the baseline prey saturate the expensive channel and economise on the cheap
> one, which is the wrong sign for a reward-scale explanation. But the ratio itself
> survives the learner fix: 2.46 of movement against 0.93 of survival, with DoS
> still at the random value under an actor that demonstrably optimises. The
> cost-zeroing cell proposed below is running.

*Claim.* With the chosen agent radii the survival term is worth ≈1 reward unit
per prey-episode while the movement budget is 6; the reward-optimal prey policy
is "stand still", not "flock", so even a perfectly working learner would leave
DoS at the random value.

*Evidence in the code.* Radii are not in the paper and are chosen at
`swarm/envs/predator_prey.py:56-59`:

```python
    # Not in the paper. 0.06 + 0.04 = 5% of the box; at max speed an agent covers
    # 0.05 m per step, half the contact diameter, so contact is always detected.
    radius_pred: float = 0.06
    radius_prey: float = 0.04
```

Ceiling on the movement cost, from `predator_prey.py:63-64, 216-217` and
Table 1 (`max_acc = 1`, `max_ang_vel = 0.5`): 0.01·1 + 0.1·0.5 = 0.06 per step =
**6.0 per episode**. Measured survival: `prey_survival_final = -1.05`, i.e.
1.05 contact-steps per prey per episode, matching `captures_final = 0.105`
(`compute_reward` counts distinct prey in contact, `predator_prey.py:194`). The
discount horizon is 1/(1−γ) = 20 steps (`swarm/algo/config.py:34`), so over the
horizon a prey weighs ~0.21 of expected capture against up to 1.2 of movement.

*What the paper says.* §3.4 defines exactly this reward and calls the movement
term "decorative", but never states the agent radii, the contact geometry, or the
resulting capture rate. §2.2 and Table 1 fix everything else. So the
survival:movement ratio — the one quantity that decides whether evasion pays — is
**entirely unconstrained by the paper** and is set here by a free parameter.

*Why it produces the observed numbers.* If evasion has negative expected value,
the optimal policy is spatially trivial (stop, or move minimally), and a trivial
policy leaves the prey distribution at its uniform reset distribution — DoS
0.2264, DoA 0.637. `npred0` is the proof that the learner *can* find that
optimum: it drops movement to −0.35 and its DoS/DoA stay flat at 0.230/0.635,
which is exactly what figure 11's n₀ = 0 control should look like. Every other
cell is a noisier version of the same non-answer.

*Confidence.* **High.** Cheap experiment: one 5-seed cell with
`cost_af = cost_ar = 0.0` (survival pressure alone), and one with
`catch_reward = 5.0`. If DoS moves off 0.226 in either, the reward balance is the
binding constraint. Note `rad30` already raises captures to 0.658/step without
helping — but `rad30` also raises the random baseline to 0.270 and floors DoS at
0.24/√2 = 0.17, so it is not a clean test of this suspect; the cost-zeroing run
is.

---

### 3. The learned predator never becomes a pursuer — and the one scripted-predator cell is the only one that shows real cohesion

*Claim.* Survival pressure is the paper's entire causal mechanism; here it is
absent, because the learned predator's capture rate never rises above the
random-motion rate. The only condition with a competent predator (`swirl`, which
uses the paper's §4.4 rule) is also the only condition where prey develop genuine
cohesion and alignment — and it was never run on the torus, so the confound with
the walls boundary has never been broken.

*Evidence in the code / on disk.* `swarm/algo/config.py:90-92` — the scripted
predator exists in exactly two presets, both walled:

```python
    "swirl": TrainConfig(env_preset="walls", normalize_obs=False, scripted_predator=True),
    "swirl_nopen": TrainConfig(env_preset="walls_nopen", normalize_obs=False,
                               scripted_predator=True),
```

`runs/exp_swirling/swirl/s0/eval_walls.json` (greedy, no exploration):

```
dos 0.0911  ->  0.0911 * 2*sqrt(2) = 0.258 m mean NN distance   (random: 0.370 m)
dos_final_quarter 0.0781  ->  0.221 m
doa 0.7735 ,  doa_final_quarter 0.8051
```

versus `runs/exp_flocking/flocking/s0/eval_torus.json` (learned predator):

```
dos 0.2143 (random 0.2264) ,  doa 0.6667 (random 0.6366)
captures_per_step 0.1045
```

The learned predator's `captures` metric is flat across the whole run
(0.101 → 0.105) and its own spatial metric never moves
(`pred_dos` 0.4156 → 0.4182 vs 0.4258 random). Plan gate 2 — "predator capture
rate rises above the scripted baseline" — is **never asserted anywhere**:
`swarm/tests/test_smoke.py:167-194` compares *scripted vs random*, never *learned
vs scripted*.

*What the paper says.* §4.4: "It is remarked that the extra boundary penalty will
complicate the environment, thus slowing down the learning process. To accelerate
the evolution of prey, we endow the predators a behavioral rule that creates
survival pressure by directly moving towards their nearest prey … By following
this rule, the predators exert survival pressure on the prey and drive their
evolutionary adaptation." §4.7 / figure 11: with n₀ = 0 "the DoS and DoA of the
prey remain unchanged, indicating that no swarming phenomenon emerges". A
predator that captures at the random-motion rate is operationally n₀ = 0.

*Why it produces the observed numbers.* If the predator exerts no pressure, every
learned cell is the n₀ = 0 control with extra noise — which is precisely what the
`exp_npredators` table shows (npred0 0.230, npred1 0.219, npred3 0.217, all
within 0.013 of each other and of the random value 0.2264).

*Confidence.* **High** on "the learned predator is incompetent", **medium** on
"that alone is why prey don't flock" (suspect 2 would still bite). Single
cheapest decisive experiment, and the one to run **first**:

```python
"swirl_torus": TrainConfig(env_preset="torus", normalize_obs=False,
                           scripted_predator=True),
```

5 seeds, ~10 min each. If DoS drops below 0.20 on the torus with a scripted
predator, the environment and the prey learner are fine and the entire problem is
the predator half of the coevolution. If it does not, suspect 2 is binding and
the reward scale has to be fixed first.

---

### 4. The turning circle is as wide as the arena — pursuit and evasion are nearly impossible in 100 steps

*Claim.* At Table 1's numbers the minimum turning radius is v/ω = 0.5/0.5 = 1.0 m
in a 2 m box, so a predator's tightest possible circle has the diameter of the
entire environment. Neither species can execute a meaningful pursuit or dodge
inside one 100-step episode, which removes most of the exploitable structure the
learner would need.

*Evidence in the code.* `swarm/envs/predator_prey.py:51-53` (`max_ang_vel = 0.5`,
`speed_pred = speed_prey = 0.5`), `swarm/envs/dynamics.py:24-30`. The repo's own
smoke test has already run into this — `swarm/tests/test_smoke.py:173-179`:

```python
    # Long episodes: max angular velocity is 0.5 rad/s, so a predator needs ~6 s
    # to turn around. Over the paper's 100 steps (10 s) aiming barely pays off.
    long = params.replace(episode_len=300)
```

i.e. the scripted rule only beats random motion once the episode is tripled.

*What the paper says.* Table 1: max speed 0.5 m s⁻¹, max angular velocity
0.5 rad s⁻¹, env edge 2 m; Table 2: episode length 100, Δt = 0.1 s. Equation
(1c): θ̇ = a_R. So the constraint is genuinely the paper's, not an implementation
error.

*Why it produces the observed numbers.* With capture events essentially
uncontrollable, the contact term in the TD target behaves as pure noise of
std ≈ 0.10 per step, while the whole action-conditional signal the critic must
resolve (the movement cost) spans only 0.05. Measured
`prey_critic_loss_final = 0.0145` ⇒ RMSE 0.12, i.e. **the critic's residual error
is 2.4× the dynamic range of the term it needs to differentiate**. That is the
concrete mechanism behind suspect 1's saturation.

*Confidence.* **Medium-high** as a contributing factor, **low** as the sole
cause. Cheap probe: a 300- or 500-step episode cell (`ep500` already exists in
`predator_prey.py:263`, and `long` shows what 20 000 × 100 gives). If DoS moves
at 500 steps but not at 100, the episode length is the binding constraint and the
paper's 100 is only workable at a capture rate we are not reproducing.

---

### 5. The perception ablation uses D = edge = 2, not D = √2 — it is 42% too generous and internally inconsistent

*Claim.* `perception_13` should be R = √2/3 = 0.471 m; it is 0.667 m. The
ablation is compressed toward "no effect", which is what `exp_perception` reports.

*Evidence in the code.* `swarm/envs/predator_prey.py:54, 261-262`:

```python
    perception: float = 2.0        # R = D = env edge in the main runs
    ...
    "perception_23": EnvParams(perception=4.0 / 3.0),
    "perception_13": EnvParams(perception=2.0 / 3.0),
```

against `swarm/envs/metrics.py:34`, which uses the other definition of D for the
same arena:

```python
    D = edge * jnp.sqrt(2.0) / (2.0 if periodic else 1.0)      # = sqrt(2) on the torus
```

*What the paper says.* Eq (2): "D ∈ R⁺ is the environment size defined as the
maximum possible distance for two agents … for a periodic square environment with
edge length 2, the largest possible distance is √2." §4.6: "predators and prey are
assumed to have a perception range equal to the environment size, that is, R = D
… we intentionally tune the perception range to be R = 2/3 D and R = 1/3 D."
Same symbol D, same paper — so R₁ = √2 ≈ 1.414, R_⅔ = 0.943, R_⅓ = 0.471.

*Why it produces the observed numbers.* At R = 0.667 on a torus whose maximum
separation is 1.414, a prey still sees most of its 6-neighbour quota, so the cell
is nearly identical to the full-perception cell — `perc13` 0.218 vs `perc33`
0.217, exactly as observed. It cannot explain the main failure, only why the
ablation looks null.

*Confidence.* **High** that it is a discrepancy, **low** that it matters for the
main result. One-line change; rerun `exp_perception`.

---

### 6. Two smaller things that bias the table

- **`reset` does not reject overlapping spawns** (`predator_prey.py:206-209`:
  `pos = uniform(-half, half)`, `vel = zeros`). At `rad30` (r = 0.18/0.12,
  13 agents in 4 m²) many pairs start overlapped and get a Hooke impulse of up to
  50·δ N on a 1 kg mass. That is why every observed "first" DoS sits ~0.01 below
  the hard-core equilibrium in the table above, and it makes the first steps of
  every radius-preset episode an explosion rather than a rollout. **Low**
  confidence of mattering, trivial to check by rendering one `rad30` episode.
- **Training-time DoS/DoA are measured with exploration on** (`act` injects
  σ = 0.05 noise and ε = 0.05 uniform actions, `swarm/algo/ddpg.py:142-150`, and
  `train_swarm.py:100` computes the metrics on the resulting state). The greedy
  eval numbers differ materially — `perc13` reports DoA 0.690 in
  `results/exp_perception.md` but `doa_final_quarter = 0.793` in
  `eval_perception_13.json`. The paper's figure 3 is also a training curve, so
  this is comparable, but the two sets of numbers in this repo are not
  interchangeable and the tracked tables should say which they are.

## Ruled out

Checked and correct — do not re-audit these.

- **DoS and DoA formulas.** `swarm/envs/metrics.py:12-42` implements eq (2) and
  eq (3) exactly: nearest *conspecific* by position under the minimum-image
  convention, self excluded via `inf` on the diagonal, `‖h_j + h_k‖/2`, and
  D = √2 on the torus per the worked example in eq (2). `metric_gates`
  (`test_smoke.py:32-67`) verifies DoA(random) = 2/π, DoA = 1 aligned, 0
  anti-aligned, and the two-opposed-flocks case that separates nearest-neighbour
  from global polarisation.
- **Metric subset and timing.** `train_swarm.py:100` takes `state.pos[prey]` /
  `state.theta[prey]` — prey only, post-step — then `train_swarm.py:136` means
  over the 100 steps. That is eq (2)'s 1/T Σ_t. Predator metrics are logged
  separately and NaN-guarded for n₀ < 2 (`train_swarm.py:53-58`).
- **`aggregate.py` windows.** `aggregate.py:56-58`: `first` = mean of episodes
  1–100, `final` = mean of the last 100, CI = 1.96·σ/√5 across the per-seed tail
  means. Correct, and the 100-episode window is the paper's own running-average
  length (§4.2, figure 3).
- **The DDPG update.** `swarm/algo/ddpg.py` is byte-identical to `a55ae17`, the
  commit where it was validated on Pendulum-v1 (`git diff a55ae17 HEAD --
  swarm/algo/ddpg.py` is empty). Adam sign (optax scales by −lr, so
  `apply_updates` on `-Q` is ascent), soft target update at τ = 0.01, actor
  updated against the freshly-updated critic, circular buffer indexing
  (cap 5e5 divisible by the 10-per-step insert, so no wrap misalignment),
  ε/noise schedule `max(0.05, x − 5e-5)` per episode — all match Algorithm 1 and
  Table 2. With `normalize_obs=False` the normaliser is the identity.
- **Bootstrapping without a done flag.** Episodes are a fixed 100 steps with no
  terminals (§3.4: "prey agents are not removed"), so treating the time limit as
  truncation and always bootstrapping is right, and the stored `next_obs` is the
  true in-episode successor — no cross-episode leakage (`train_swarm.py:73-75`,
  `126-130`).
- **Observation construction.** 54 = 6 + 2·6·4 (`predator_prey.py:72-73`), i.e.
  own pos/vel/heading plus ≤6 allies and ≤6 adversaries × (rel pos, heading).
  `_gather` (`predator_prey.py:123-137`) sorts nearest-first, masks beyond R,
  zero-pads, and excludes self; `env_gates` asserts all four properties. Matches
  §2.3 and Appendix B. (The paper never states d_o numerically — 54 follows from
  the repo's unit-vector heading choice, which is a documented deviation.)
- **Integration order and physics.** `dynamics.py:23-32` does θ then v (with h
  from the *new* θ) then x (with the *old* v), exactly Appendix B. Drag 2 gives a
  terminal speed of a_F/drag = 0.5 = the Table 1 speed cap, so the two are
  consistent. Torus wrap, minimum image, no tunnelling and contact resolution are
  all gated in `env_gates` (`test_smoke.py:99-133`).
- **Reward semantics.** Computed on the post-step state with the action actually
  executed; prey are not removed; contact bleeds −1 every step and returns to
  zero on separation; boundary penalty only in walls mode. Matches §3.4.
- **Seed vmapping and the freeze bookkeeping.** `train.py:85-92` vmaps distinct
  `PRNGKey(s)` and slices result index `i` into leaf `s{i}`; the
  `chunk`/`_flat` reshapes in `train_swarm.py:140-174` merge (outer, inner) in
  the right order, and `freeze_gate` verifies that a frozen species takes no
  gradient step.
- **The n₀ = 0 control.** 0.229 → 0.230 / 0.626 → 0.635 is flat, which is what
  figure 11 requires. It is not evidence of a bug; it is the one cell that is
  currently *right*.

## Open questions the paper genuinely leaves undetermined

1. **Agent radii / contact geometry.** Never stated. It alone sets the capture
   rate and therefore the survival:movement ratio — the quantity suspect 2 turns
   on. Everything else in Table 1 is pinned; this is the free parameter that
   decides whether the paper's reward is learnable.
2. **Whether −0.01|a_F| − 0.1|a_R| is charged on the physical action or on the
   network's [−1,1] output.** The repo charges the physical one
   (`predator_prey.py:186-187`), so |a_R| ≤ 0.5 and the rotation term maxes at
   0.05/step. On the normalised output it would max at 0.1/step — twice as
   punishing, same qualitative conclusion.
3. **Initial velocity.** §"Appendix B" says "random positions with random
   headings" and says nothing about v. The repo uses zero
   (`predator_prey.py:207`).
4. **Whether the soft target update is per step or per episode.** Algorithm 1's
   indentation places it in the episode loop, one line outside the step loop; the
   repo does it per step, which is standard MADDPG.
5. **"Relative pos. and headings of observed predators" (Appendix B)** — is the
   neighbour's heading absolute or expressed in the observer's frame? The repo
   uses absolute unit vectors (`predator_prey.py:135-136`).
6. **Observation ordering.** Appendix B fixes predators-then-prey for both
   species; the repo uses allies-then-adversaries (documented deviation,
   `predator_prey.py:141-145`). Harmless with per-species actors, but it means
   the two species do not share the paper's input semantics.
7. **R = D for perception: √2 or 2?** See suspect 5. The paper uses one symbol D
   for both; eq (2) defines it as √2.
8. **Which population figure 4 uses.** §4.1 says evaluation deploys 50 prey, but
   figure 4's DoS ≈ 0.15 is impossible at n₁ = 50 (random is 0.100), so figure 4
   must be n₁ = 10. Worth stating explicitly in any writeup, because DoS is not
   density-normalised and cross-population comparisons are meaningless.
