# exp_learner_sweep — 2026-08-31

**Superseded by `exp_sweeps.md` (2026-09-01). Kept for its narrative. Three claims
below are now known to be wrong:**

1. *"why formation is 5–10× slower than the paper's"* — **wrong.** Measured on
   single-episode curves, DoA crosses 0.9 at step 43–79 and DoS reaches 0.13 by
   step 50–100, comparable to the paper's fig 4. The slow ramp was an artifact of
   averaging episodes that form at different moments. The real defect is that the
   flock **breaks and re-forms**, not that it forms slowly.
2. *"`n_envs` 64: no effect on flocking. Removes 'the learner is data-starved' as
   an explanation."* — **not supported.** `n_envs` multiplies data collected but
   leaves the number of gradient steps unchanged, so the learner does exactly as
   much learning either way. The experiment cannot test data-starvation.
3. The handling-time and Tier-1 results described here were later re-measured;
   see `exp_sweeps.md`. `handling_time` was removed from the code on 09-01.

Handoff note. Read this before touching the flocking replication again.

## The headline

**Our prey do flock, significantly, and I spent two days claiming they did not.**
The claim was based on the training-logged DoS/DoA, which are episode means over
100 steps starting from a uniform random spawn. Most of that window is the
formation transient. The flock is there; the metric hides it.

Under evaluation, learned prey vs random prey, same trained predators, 200
episodes × 5 seeds, `episode_len` 300, greedy `mu(o)`, paired over seeds:

| | DoS final ¼ | DoA final ¼ |
|---|---|---|
| random prey | 0.2220 ± 0.0014 | 0.6414 ± 0.0012 |
| learned prey | 0.2071 ± 0.0032 | 0.7274 ± 0.0339 |
| paired difference | **−0.0149, t = −7.72** | **+0.0860, t = +5.14** |

df = 4, threshold 2.78. Both significant.

The effect grows with the observation window — DoA advantage over random is
+0.025 at L=100, +0.055 at L=300, +0.086 in the final quarter of L=300.

**Exploration noise is not the cause.** Greedy eval at L=100 gives DoS 0.2218 /
DoA 0.6641, which matches the training log (0.2235 / 0.6634) almost exactly. It
is the episode length alone.

## What the real gap is

On the paper's own metric at the paper's own L=100 we are still short: 0.2218 vs
0.19. But the paper's fig 4 shows their DoS reaching ~0.15 **by about step 30**.
Ours needs 200+ steps to reach 0.207.

The open question is no longer *whether* the flock forms. It is **why formation
is 5–10× slower than the paper's**. Everything below should be read in that light.

Alignment leads cohesion throughout. In the best episode DoA is 0.79 by step 60
while DoS barely moves until step 225. Turning together is cheap (`cost_ar`);
closing is expensive (accelerate against drag and hold station).

## Runs

All on the semi-implicit physics, 5 seeds, 3v10, 2000 × 100 unless noted.
Reference is `flocking_reg` = `flocking.json` + `actor_reg = 0.001`.

| run dir | config | change from reference |
|---|---|---|
| `20260830-194629_2c723019` | `flocking_reg` | — (the reference) |
| `20260831-105755_20065180` | `flocking_envs64` | `n_envs` 1 → 64 |
| `20260831-141526_7fe0799a` | `flocking_shortbuf` | `buffer_size` 5e5 → 8000, `n_envs` 1 |
| `20260831-142953_9fdef60b` | `flocking_gamma99` | `gamma` 0.95 → 0.99, `n_envs` 64 |
| `20260831-145707_c7ed7ac7` | `flocking_explore` | `expl_min` 0.05 → 0.15, `n_envs` 64 |
| `20260831-152445_f9404230` | `flocking_batch1024` | `batch_size` 256 → 1024, `n_envs` 64 |
| `20260831-160737_4c9eefe4` | `flocking_long300` | `episode_len` 100 → 300, `n_envs` 64 |

Each differs from its base in exactly one field.

### Training metrics, last 200 episodes

| run | DoS | DoA | captures/step | prey return | prey Q | critic loss |
|---|---|---|---|---|---|---|
| baseline 1 env | 0.2235 ± 0.0024 | 0.6634 ± 0.0088 | 0.0747 | −3.21 | −0.53 | 0.011 |
| shortbuf 1 env | 0.2215 ± 0.0033 | 0.6645 ± 0.0042 | 0.1219 | −3.88 | −0.65 | 0.012 |
| envs64 | 0.2215 ± 0.0017 | 0.6621 ± 0.0106 | 0.1164 | −3.41 | −0.58 | 0.024 |
| + gamma99 | 0.2203 ± 0.0023 | 0.6585 ± 0.0087 | 0.0842 | −2.70 | −2.13 | 0.026 |
| + explore | 0.2229 ± 0.0006 | 0.6613 ± 0.0012 | 0.1558 | −4.32 | −0.65 | 0.022 |
| + batch1024 | 0.2196 ± 0.0013 | 0.6803 ± 0.0144 | 0.1216 | −4.06 | −0.71 | 0.024 |
| + long300 | 0.2144 ± 0.0040 | 0.7135 ± 0.0295 | 0.1710 | −10.44 | −0.55 | 0.033 |

`long300`'s apparent gain is an artifact: its episodes are 3× longer, so the
transient is a smaller fraction of the mean. At a matched 300-step evaluation
window it is **not** better than the baseline (DoA +0.052, t = 0.94; DoS −0.0015,
t = −0.31).

### Verdicts

- **`n_envs` 64: no effect on flocking.** 64× the data changed nothing in DoS/DoA.
  Removes "the learner is data-starved" as an explanation.
- **`shortbuf` was the useful control.** A 5e5 buffer at 64 envs spans only ~8
  episodes (640 rows/step). Reproducing that span at 1 env (8000 rows) gives
  captures 0.1219 — matching envs64's 0.1164. **The predator improvement I
  credited to 64 envs was recency, not data volume.**
- **`gamma99`: better evasion, no flocking change.** Captures down to 0.0842,
  best prey return (−2.70). The 2-second-horizon argument did not pay off in
  DoS/DoA.
- **`explore`: pure harm.** Worst prey return, most captures.
- **`batch1024`: nothing.**
- **`long300`: nothing, once measured correctly.**

## The correct instrument

Do not read DoS/DoA off `metrics.npz` and compare to the paper. Use
`run/eval.py` with a matched window and a **random-prey control at the same
predator pressure**. Eval configs are under `eval_configs/matched/`, results in
`evals/matched_*.json`.

The controls that matter: `matched_preyrandom_s{0..4}` and
`matched_preyuntrained_s{0..4}` — prey random/untrained, predators learned from
the reference run. Untrained prey behave like random (DoS 0.2244, DoA 0.6451).

DoS is **not density-normalised** (nearest-neighbour distance falls as 1/√N), so
it cannot be compared across prey counts. Random-configuration DoS in the
edge-2 torus: **N=10 → 0.2271, N=30 → 0.1297, N=50 → 0.1003**.

## Best policy found

`runs/20260831-160737_4c9eefe4` seed 3 (`flocking_long300`). Renders exist for it:

| render | config | DoS mean | DoA mean | final ¼ |
|---|---|---|---|---|
| `renders/bestflock` | 3v10, L=300 | 0.2043 | 0.8835 | 0.1810 / 0.9618 |
| `renders/bestflock500` | 3v10, L=500 | 0.1912 | 0.8676 | 0.1824 / 0.7910 |
| `renders/bestflock500_p6` | 6v10, L=500 | 0.1958 | 0.7982 | 0.2144 / 0.9213 |
| `renders/bestflock500_p9` | 9v10, L=500 | 0.1970 | 0.8361 | 0.1940 / 0.9493 |
| `renders/bestflock500_n30` | 3v30, L=500 | 0.1057 | 0.9159 | 0.1015 / 0.9652 |
| `renders/bestflock500_n50` | 3v50, L=500 | 0.0871 | 0.8822 | 0.0755 / 0.9330 |

3v50 is the paper's evaluation config (§4.1: train 3v10, deploy on 50 prey, same
predator count).

Two findings from these:

- **The 3v10 flock is unstable.** At L=500 it reaches DoS 0.141 / DoA 0.923 at
  steps 350–400, better than the paper's fig 4 floor on sparsity, then breaks at
  step 400 (DoA 0.92 → 0.75) and does not re-form. The paper's flock reaches its
  floor and stays.
- **More agents stabilise it.** At 3v30 and 3v50, and at 6v10 and 9v10, the final
  quarter holds DoA 0.92–0.97. With only 10 prey, one scatter event destroys the
  group; with more, there is always a core to re-form around.

The policy was trained at 3v10 and generalises to 9 predators and 50 prey unseen,
because the observation is topological (6 nearest of each species).

## Code changes this session

- `config.py`: `MADDPGConfig.n_envs`. Backfilled `"n_envs": 1` into the 4 tracked
  configs and all 11 `runs/*/config.json` (every earlier run was single-env).
- `algo/maddpg.py`: `reset`/`step`/`reward`/`observe`/`scale_action`/
  `scripted.predator` vmapped over an env axis; `act` sizes its noise from
  `obs.shape[:-1]`; the buffer has no env axis, so a step reshapes
  `(n_envs, n_i, d)` into rows; DoS/DoA/captures averaged across envs before logging.
- `tests/test_maddpg.py`: `test_parallel_envs_keep_the_metric_shape`.
- `run/eval.py` and `eval_configs/matched/`, `eval_configs/bestflock*.json`.

`test_env`, `test_networks`, `test_maddpg` all pass.

**Note on `n_envs` > 1**: the buffer is not vmapped, so insertion rate scales with
`n_envs` and the buffer's *episode span* divides by it. At 64 envs a 5e5 buffer
holds ~8 episodes of prey experience instead of ~500. Raising `buffer_size` to
compensate is not affordable on CPU — buffers are vmapped over seeds, and 5e5
rows is already ~2.2 GB across 2 species × 5 seeds.

## Next

1. **Log `dos_final_quarter` / `doa_final_quarter` in training.** The episode mean
   is the wrong summary and every run so far has been read through it. Ten lines
   in `episode()` in `maddpg.py`.
2. **Re-read the earlier screens with the corrected window.** The handling-time
   result (3.2% at k=20) and the whole Tier-1 table were measured over 100-step
   episodes dominated by the transient. They may be badly understated.
3. **Attack formation speed, not existence.** That is the actual remaining gap.
4. **Stability at small N** is a separate, real phenomenon worth its own note.

## Stale notes — do not trust

- `replication_gap_suspects.md` — deleted 09-01; it predated the integrator fix.
  Recover from git history if a detail is ever needed.
- `inducing_flocking.md` — rewritten 09-01. Its old Tier-1 table was old-physics
  and read through the wrong window; the surviving screen results are in
  `exp_sweeps.md`.
