# Tables

Numbers only. Interpretation is in `exp_sweeps.md`. Regenerated 2026-09-02.

## 1. Training runs

`runs/<dir>/s*/metrics.npz`, `[-200:].mean()` per seed, then mean ± 2.78 × SEM
across the 5 seeds. `preyR` / `surv` / `move` are per-episode prey totals.

**Comparable only within one L.**

| run_dir | config | L | DoS | DoA | cap | preyR | surv | move |
|---|---|---|---|---|---|---|---|---|
| 20260829-155023_28ffe830 | flocking | 100 | 0.2211 ±.0066 | 0.6743 ±.0160 | 0.1059 | −5.73 | −1.06 | −4.67 |
| 20260830-114547_9ebf0e65 | flocking_reg ⟨old⟩ | 100 | 0.2237 ±.0023 | 0.6687 ±.0128 | 0.0964 | −3.42 | −0.96 | −2.46 |
| 20260830-124249_fdd0c47a | flocking_nocost ⟨old⟩ | 100 | 0.2220 ±.0035 | 0.6669 ±.0176 | 0.0953 | −0.95 | −0.95 | 0.00 |
| 20260830-131828_02519184 | flocking_scripted ⟨old⟩ | 100 | 0.2258 ±.0031 | 0.6519 ±.0111 | 0.1120 | −3.74 | −1.12 | −2.62 |
| 20260830-144159_2c723019 | flocking_reg ⟨old⟩ | 100 | 0.2256 ±.0015 | 0.6546 ±.0183 | 0.1017 | −3.33 | −1.02 | −2.31 |
| 20260830-145636_585e3787 | flocking_nocost ⟨old⟩ | 100 | 0.2212 ±.0021 | 0.6711 ±.0173 | 0.0966 | −0.97 | −0.97 | 0.00 |
| 20260830-151114_da6cbad6 | flocking_scripted ⟨old⟩ | 100 | 0.2256 ±.0036 | 0.6582 ±.0113 | 0.1144 | −3.81 | −1.14 | −2.66 |
| **20260830-194629_2c723019** | **flocking_reg — REFERENCE** | 100 | 0.2235 ±.0033 | 0.6634 ±.0122 | 0.0747 | −3.21 | −0.75 | −2.46 |
| 20260830-200111_da6cbad6 | flocking_scripted | 100 | 0.2220 ±.0026 | 0.6615 ±.0073 | 0.1118 | −3.84 | −1.12 | −2.72 |
| 20260831-105755_20065180 | flocking_envs64 | 100 | 0.2215 ±.0024 | 0.6621 ±.0147 | 0.1164 | −3.41 | −1.16 | −2.25 |
| 20260831-141526_7fe0799a | flocking_shortbuf | 100 | 0.2215 ±.0046 | 0.6645 ±.0058 | 0.1219 | −3.88 | −1.22 | −2.66 |
| 20260831-142953_9fdef60b | flocking_gamma99 | 100 | 0.2203 ±.0031 | 0.6585 ±.0120 | 0.0842 | −2.70 | −0.84 | −1.86 |
| 20260831-145707_c7ed7ac7 | flocking_explore | 100 | 0.2229 ±.0008 | 0.6613 ±.0017 | 0.1558 | −4.32 | −1.56 | −2.76 |
| 20260831-152445_f9404230 | flocking_batch1024 | 100 | 0.2196 ±.0017 | 0.6803 ±.0200 | 0.1216 | −4.06 | −1.22 | −2.85 |
| 20260831-160737_4c9eefe4 | flocking_long300 | 300 | 0.2144 ±.0055 | 0.7135 ±.0410 | 0.1710 | −10.44 | −5.13 | −5.31 |
| 20260901-230643_16e8b80b | flocking_long300_gamma99 | 300 | 0.2145 ±.0048 | 0.7407 ±.0740 | 0.1488 | −9.18 | −4.46 | −4.71 |
| 20260901-161634_ad74c0b7 | flocking_long500 | 500 | 0.2012 ±.0075 | 0.8265 ±.0744 | 0.1539 | −21.43 | −7.69 | −13.74 |
| 20260902-101053_ce34ab2d | flocking_long300_nocost | 300 | 0.2001 | 0.8242 | 0.1327 | −3.98 | −3.98 | 0.00 |

⟨old⟩ = broken forward-Euler integrator, not comparable to anything below it.
`flocking_long300_nocost` is **seed 0 only** — no CI, no significance. Its paired
control is `flocking_long300` **seed 0**, not that run's 5-seed mean:

| L=300, seed 0 | DoS | DoA | cap | preyR | prey_af | prey_ar | pred_af |
|---|---|---|---|---|---|---|---|
| flocking_long300 | 0.2193 | 0.6769 | 0.1650 | −8.75 | 0.1457 | 0.1123 | 0.4024 |
| flocking_long300_gamma99 | 0.2186 | 0.6929 | 0.1192 | −7.07 | 0.1549 | 0.1008 | 0.3311 |
| flocking_long300_nocost | 0.2001 | 0.8242 | 0.1327 | −3.98 | 0.3288 | 0.2760 | 0.7002 |

DoA per 200 episodes across training, seed 0 — `nocost` has not converged:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| long300 | 0.712 | 0.711 | 0.726 | 0.793 | 0.797 | 0.685 | 0.685 | 0.693 | 0.696 | 0.677 |
| nocost | 0.725 | 0.708 | 0.722 | 0.722 | 0.695 | 0.695 | 0.710 | 0.722 | 0.778 | 0.824 |
`20260829-145907_28ffe830` has no `metrics.npz` — aborted run.
`20260829-174936_28ffe830` is bit-identical to `20260829-155023` (same config,
same seeds, deterministic).

## 2. Evals

Each row = 5 files `evals/<cell>_s0.json` … `_s4.json`, 200 episodes each,
3v10 unless the cell says otherwise. Prey only; predators are sliced off.

DoS and DoA are computed per prey per step (`metrics.dos` / `metrics.doa`, eq 2
and 3), giving a (200, L) array per file, then flat-averaged over every value.
`/4` columns use only the last quarter of steps. The table averages the five
files' `mean` fields.

Values behind one row: 5 × 200 × L steps × 10 prey.

| cell | L | n | DoS | DoA | DoS/4 | DoA/4 | speed | cap | preyR |
|---|---|---|---|---|---|---|---|---|---|
| matched_l100rand | 100 | 5 | 0.2263 | 0.6392 | 0.2225 | 0.6406 | — | 0.2505 | −5.50 |
| matched_l100base | 100 | 5 | 0.2218 | 0.6641 | 0.2134 | 0.6818 | — | 0.0650 | −3.15 |
| matched_preyrandom | 300 | 5 | 0.2235 | 0.6424 | 0.2220 | 0.6414 | — | 0.2993 | −17.98 |
| matched_preyuntrained | 300 | 5 | 0.2244 | 0.6451 | 0.2225 | 0.6510 | — | 0.2940 | −12.07 |
| matched_base | 300 | 5 | 0.2129 | 0.6978 | 0.2071 | 0.7274 | — | 0.0745 | −9.59 |
| matched_envs64 | 300 | 5 | 0.2147 | 0.6924 | 0.2077 | 0.7203 | — | 0.2340 | −13.32 |
| matched_long300 = m300_l300b | 300 | 5 | 0.2144 | 0.7294 | 0.2056 | 0.7796 | 0.0840 | 0.2134 | −10.99 |
| m300_g99 | 300 | 5 | 0.2126 | 0.7814 | 0.2020 | 0.8564 | 0.0766 | 0.1597 | −8.88 |
| m500_rand | 500 | 5 | 0.2239 | 0.6404 | 0.2236 | 0.6394 | 0.2437 | 0.6701 | −48.51 |
| m500_long300 | 500 | 5 | 0.2086 | 0.7555 | 0.1987 | 0.7963 | 0.0856 | 0.2862 | −21.48 |
| m500_long500 | 500 | 5 | 0.2035 | 0.8304 | 0.1925 | 0.8597 | 0.0853 | 0.1775 | −22.06 |
| nopred_rand | 500 | 5 | 0.2247 | 0.6368 | 0.2258 | 0.6312 | 0.2445 | 0 | −15.00 |
| nopred_long300 | 500 | 5 | 0.2057 | 0.8009 | 0.1881 | 0.8533 | 0.4092 | 0 | −7.90 |
| nopred_long500 | 500 | 5 | 0.2150 | 0.7452 | 0.2081 | 0.7845 | 0.3920 | 0 | −8.45 |

`speed` is mean prey speed, cap 0.5. Blank where the cell predates the
`prey_speed` field. `nopred_*` cells have `n_pred: 0`.

Which run each cell loads:

| cell | predator | prey |
|---|---|---|
| matched_base, matched_l100base | 20260830-194629_2c723019 | same |
| matched_preyrandom / preyuntrained / l100rand | 20260830-194629_2c723019 | random / untrained |
| matched_envs64 | 20260831-105755_20065180 | same |
| matched_long300, m300_l300b, m500_long300, nopred_long300 | 20260831-160737_4c9eefe4 | same |
| m300_g99 | 20260901-230643_16e8b80b | same |
| m500_long500, nopred_long500 | 20260901-161634_ad74c0b7 | same |
| m500_rand, nopred_rand | 20260901-161634_ad74c0b7 | random |

## 3. Paired tests

Paired over the 5 seeds, df = 4, `*` = |t| > 2.78.

| comparison | ΔDoS/4 | t | ΔDoA/4 | t | Δcap | t | ΔpreyR | t |
|---|---|---|---|---|---|---|---|---|
| matched_long300 − matched_base | −0.0015 | −0.31 | +0.0521 | +0.94 | +0.1389 | **+4.89\*** | −1.397 | −2.23 |
| m300_g99 − m300_l300b | −0.0036 | −0.57 | +0.0768 | +1.23 | −0.0537 | **−3.00\*** | +2.108 | **+3.02\*** |
| m500_long500 − m500_long300 | −0.0062 | −0.82 | +0.0634 | +1.01 | −0.1087 | **−3.14\*** | −0.577 | −0.34 |
| m500_long500 − m500_rand | −0.0311 | **−8.92\*** | +0.2203 | **+4.82\*** | −0.4926 | **−4.55\*** | +26.45 | **+5.60\*** |
| m500_long300 − m500_rand | −0.0249 | **−3.91\*** | +0.1569 | **+3.40\*** | −0.3840 | **−3.09\*** | +27.03 | **+4.43\*** |
| nopred_long500 − nopred_rand | −0.0176 | −1.82 | +0.1533 | **+4.14\*** | 0 | — | +6.55 | **+18.15\*** |
| nopred_long300 − nopred_rand | −0.0377 | −2.34 | +0.2221 | **+4.75\*** | 0 | — | +7.10 | **+17.36\*** |

## 4. Formation screens

Capture rate at the tightest spacing ÷ at the loosest. Below 1.0 = grouping pays.
Prey policy `20260830-194629_2c723019`, scripted predator, L=100, lattice spawn,
spacings 0.15–0.45 (tier1 also 0.10), averaged over prey seeds `_p0..p4`.
Source field: `captures_per_step.mean`.

| screen | cell | ratio |
|---|---|---|
| tier1 | base | 1.19 |
| tier1 | npred1 | 1.13 |
| tier1 | npred5 | 1.09 |
| tier1 | npred8 | 1.18 |
| tier1 | fast_pred | 1.14 |
| tier1 | fast_prey | 1.27 |
| tier1 | nprey50 | 1.10 |
| tier1 | small | 1.02 |
| tier1b | npred1 | 1.02 |
| tier1b | npred5 | 1.08 |
| tier1b | nprey50 | 1.04 |
| tier1b | small | 1.03 |
| agility | turn1 | 1.03 |
| agility | turn2 | 1.07 |
| agility | turn4 | 1.09 |
| agility | turn8 | 1.05 |
| agility | fast | 1.08 |
| agility | fastturn | 1.20 |
| preyspeed | base | 1.03 |
| preyspeed | paper35 | 1.22 |
| preyspeed | prey15x | 1.04 |
| preyspeed | prey2x | 1.05 |
| handling | k00 | 1.03 |
| handling | k03 | 1.04 |
| handling | k05 | 1.02 |
| handling | **k10** | **0.99** |
| handling | **k20** | **0.97** |

`handling_time` was removed from the code on 09-01 (before commit `4c0218d1`).
The `evals/handling_*.json` results cannot be regenerated.

## 5. Within-episode curves

One episode, 3v10, L=500, `env_seed` 0, each run's best seed (`long300` s3,
`long500` s2). From `eval_configs/curve_long300.json` / `curve_long500.json` →
`renders/curve_*/traj.npz`.

| step | 0 | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 499 |
|---|---|---|---|---|---|---|---|---|---|---|
| long300 DoS | 0.201 | 0.246 | 0.209 | 0.148 | 0.264 | 0.207 | 0.229 | 0.158 | 0.172 | 0.133 |
| long300 DoA | 0.525 | 0.642 | 0.826 | 0.879 | 0.899 | 0.955 | 0.971 | 0.976 | 0.979 | 0.665 |
| long500 DoS | 0.202 | 0.159 | 0.136 | 0.132 | 0.115 | 0.138 | 0.168 | 0.219 | 0.123 | 0.219 |
| long500 DoA | 0.514 | 0.670 | 0.934 | 0.956 | 0.980 | 0.985 | 0.943 | 0.843 | 0.985 | 0.961 |
| nocost DoS | 0.202 | 0.205 | 0.170 | 0.181 | 0.226 | 0.209 | 0.145 | 0.250 | 0.199 | 0.144 |
| nocost DoA | 0.523 | 0.464 | 0.403 | 0.566 | 0.678 | 0.883 | 0.798 | 0.682 | 0.927 | 0.971 |

`nocost` row is `eval_configs/nocost_500_n10.json` → `renders/nocost_500_n10`,
seed 0 (the only seed trained).

| | DoA first >0.9 | min DoS | last ¼ DoS | last ¼ DoA | prey speed | pred speed |
|---|---|---|---|---|---|---|
| long300 | step 79 | 0.098 @ 70 | 0.182 | 0.791 | 0.129 | 0.260 |
| long500 | step 43 | 0.095 @ 389 | 0.178 | 0.978 | — | — |
| nocost | step 153 | — | 0.169 | **0.908** | 0.117 | 0.318 |

`long300` peaks at DoA 0.979 by step 400 then collapses to 0.665 by 499.
`nocost` forms slower but does not collapse.

## 6. Paper targets

From `li2023_spec.md` C2–C5 (§4.2, figs 3–4), all at their L=100.

| | paper | us L=100 | us L=500 |
|---|---|---|---|
| DoS, random start | 0.22 | 0.226 | 0.224 |
| DoA, random start | 0.65 | 0.639 | 0.640 |
| DoS, trained | 0.19 | 0.222 | 0.204 |
| DoA, trained | 0.82 | 0.664 | 0.830 |
| DoS, within-episode floor | 0.15 | — | 0.193 (last ¼) |
| DoA, within-episode peak | 0.96 | — | 0.860 (last ¼) |

## Notes on reading any of this

- **DoS is not density-normalised.** Nearest-neighbour distance falls as 1/√N, so
  DoS cannot be compared across prey counts. Random-configuration DoS on the
  edge-2 torus: N=10 → 0.227, N=30 → 0.130, N=50 → 0.100.
- The `sd` field inside `evals/*.json` is spread **across episodes**. The ± in
  table 1 is 2.78 × SEM **across seeds**. Different quantities; do not mix.
- Training-log DoS/DoA are episode means, so they are only comparable within one
  `episode_len`. Cross-`L` comparison must go through `run.eval`.
