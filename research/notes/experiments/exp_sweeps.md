# Part 1 — every experiment we ran

There are only two kinds. Training = the learner runs, weights change, a folder
appears in `runs/`. Eval = nothing learns, a frozen policy is loaded and
measured, a file appears in `evals/`.

## Training runs

All of them: MADDPG, 3 predators vs 10 prey, torus arena edge 2.0, 2000
episodes, 5 seeds, 100 steps per episode. Only the listed field differs.

| when | config | what changed | physics |
|---|---|---|---|
| 08-29 | `flocking` ×3 | nothing — the baseline | old |
| 08-30 am | `flocking_reg` | added `actor_reg` 0.001 | old |
| 08-30 am | `flocking_nocost` | movement cost set to 0 | old |
| 08-30 am | `flocking_scripted` | predator is the fixed rule, only prey learn | old |
| | *(those three were run twice, before and after a logging change)* | | |
| 08-30 pm | `flocking_reg` | same config, rerun after the integrator fix | new |
| 08-30 pm | `flocking_scripted` | same, rerun | new |
| 08-31 | `flocking_envs64` | `n_envs` 1 → 64 | new |
| 08-31 | `flocking_shortbuf` | `buffer_size` 5e5 → 8000, 1 env | new |
| 08-31 | `flocking_gamma99` | `gamma` 0.95 → 0.99 | new |
| 08-31 | `flocking_explore` | `expl_min` 0.05 → 0.15 | new |
| 08-31 | `flocking_batch1024` | `batch_size` 256 → 1024 | new |
| 08-31 | `flocking_long300` | `episode_len` 100 → 300 | new |

The last five all inherited `n_envs: 64` from `envs64`.

Two runs matter downstream:

- **`flocking_reg` (08-30 evening)** — the reference policy. Every eval screen
  loads its prey.
- **`flocking_long300`, seed 3** — the best policy found. Every GIF comes from it.

## Eval screens

### Group A — the formation test

**Question:** if you put the prey in a tight group, do fewer of them get eaten?

Setup, identical across all four: prey placed on a grid at a chosen spacing,
predator is the fixed scripted rule, prey policy loaded frozen from
`flocking_reg`, 100 steps, no learning. Spacings from 0.15 (packed) to 0.45
(spread).

| when | screen | what was varied | answer |
|---|---|---|---|
| 08-30 | `tier1`, `tier1b` | predator count, speed ratio, prey count, arena size | no — 1.02 to 1.19, tight is always worse |
| 08-31 | `agility` | predator turn rate ×1 to ×8, predator acceleration | no — 1.03 to 1.20 |
| 08-31 | `preyspeed` | prey speed ×1, ×1.5, ×2 | no — 1.03 to 1.22 |
| 08-31 | `handling` | predator frozen 0/3/5/10/20 steps after a catch | yes at 10 and 20 — 0.99 and 0.97 |

`tier1` and `tier1b` are the same screen; `tier1b` repeated the surviving cells
across 5 prey seeds.

### Group B — the matched eval

**Question:** do trained prey flock more than random prey do?

Setup: normal random spawn, 300 steps, both species loaded frozen, and a
random-prey control run at the same predator pressure. 08-31 afternoon.

**Answer:** yes. Trained prey reach DoA 0.73 in the final quarter against 0.64
for random. The earlier "they don't flock" conclusion came from measuring
100-step episodes, which are almost entirely the prey still gathering.
