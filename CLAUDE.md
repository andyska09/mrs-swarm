# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT NOTE TO ALL AI CHATBOTS

When talking to me start the message with "TARS:"

### Style

- Blunt, minimal code. No base classes, protocols, registries. One file per
  concern, readable top to bottom.
- **Comments: as few as possible.** Only where the code cannot say it itself —
  a constraint, a non-obvious reason, a trap. Never restate what the line does.
  One clear sentence beats three. No banner blocks, no section dividers.
- Same for docstrings: a module gets a short one, a function only if its
  contract is not obvious from its name and signature.

**Answer the question I actually asked. Nothing more.** If I ask what a term means, explain
the term - do not re-analyse the plan, do not revise your recommendations, do not rewrite
your conclusions. A question is not a signal that you were wrong. When I hand you new
information, absorb it and answer; only redo the analysis if I explicitly tell you to. If
something I said genuinely changes an earlier conclusion, say so in one line at the end and
stop there - wait for me to ask before expanding it.

I am new to this field. When I ask about a term, assume I want the concept explained plainly,
not a literature review.

When I shout and curse at you DO NOT apologize, it wastes tokens, just follow orders. 
Next when writing code DO NOT write stupid comments and docstrings. Keep the code clean and high quality. When implementing stuff KEEP IT SIMPLE. This important. I do not want to read 1000 lines of diffs, I want to look at the change and know what it does. 

#### FOR CODEX:

Edit only the named file. Do not run tests. Stop after the edit, summarize it, and wait for approval before touching another file. Unless I say to you to run wild.

## What this repository is

A research workspace for multi-robot / swarm RL. Our own code lives
in `swarm_simple/`; `research/` holds notes, papers, and two vendored reference
codebases we learn from but do not edit.

Replicating Li 2023 is how the platform gets validated; the platform is the end.

Project goals ([research/notes/goals.local.md](research/notes/goals.local.md),
written in Czech — gitignored, so a fresh clone does not have it):

1. Replicate *"Predator–prey survival pressure is sufficient to evolve swarming
   behaviors"* (Li, Li & Zhao 2023, New J. Phys. 25 092001 — in `research/papers/`).
   MADDPG coevolution with a purely survival-based reward; swarming emerges
   without swarm-specific shaping.
2. Use the supervisor's PPO project (`research/code_sources/PPO_example`) as the
   learner baseline.
3. Study swarming with attention via
   [quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl).
4. Push the paper further: add attention, a 3D environment, and/or more
   sophisticated dynamics.

## Python environments. There are two, they are not compatible, never mix them.

| env | for | how |
|---|---|---|
| `mrs-swarm` | `swarm_simple/`, `research/code_sources/PPO_example` | `conda run -n mrs-swarm --no-capture-output python ...` |
| — | `research/code_sources/quad-swarm-rl` | separate PyTorch stack, not created yet |

**NEVER run bare `python` or `python3`.** The system interpreter has no jax. Every
invocation goes through `conda run -n mrs-swarm --no-capture-output`, including
one-off checks and test scripts.

`mrs-swarm` is conda + python 3.11, CPU: jax/jaxlib 0.10.1, flax 0.12.7, optax
0.2.8, matplotlib, imageio ([requirements.txt](requirements.txt), pins copied
from PPO_example). Use `conda run`, not `source activate`. Run everything from
the repo root as `-m swarm_simple.…`.

## swarm_simple/ — the only code

Ground-up reimplementation, CleanRL style. It replaced an earlier tree, `swarm/`,
deleted in `99daf31` — recover it from git history if a detail is ever needed.

Four references, in this order:
[li2023_spec.md](research/notes/li2023_spec.md) is the paper side only;
[choices.md](research/notes/choices.md) is every decision the paper does not make
for us, and what we hardcode where it does;
[exp_sweeps.md](research/notes/experiments/exp_sweeps.md) is every run and eval
we have done and what each one measured;
[exp_learner_sweep.md](research/notes/experiments/exp_learner_sweep.md) is the
current state of the replication — read it before touching flocking again.
[inducing_flocking.md](research/notes/inducing_flocking.md) is the ranked list of
what to try next.

Deviations from the paper, all deliberate: agent radii (0.06 / 0.04) and initial
velocity are unstated in the paper; observations use ally-first ordering and
unit-vector headings; `pred_acc_scale` / `pred_turn_scale` / `max_speed_*` are
knobs the paper does not have (1.0 is the paper); `spawn: lattice` exists only
for the formation test.

```bash
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.run.train configs/flocking.json --seeds 0 1 2
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.run.replay eval_configs/flock50.json
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.run.render renders/flock50
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.run.eval eval_configs/flock50.json --episodes 200
```

Tests: **no pytest, and no aggregate gate.** Each module is its own `__main__`
runner and is invoked on its own; nothing runs them all.

```bash
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.tests.test_env      # ~450 lines, the big one
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.tests.test_metrics
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.tests.test_networks
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.tests.test_buffer
conda run -n mrs-swarm --no-capture-output python -m swarm_simple.tests.test_maddpg   # runs tiny trainings, slowest
```

```
swarm_simple/
├── config.py                the ONLY schema: frozen dataclasses + load/hash
├── envs/predator_prey.py    state, step, observe, reward, rollout — module of functions
├── envs/metrics.py          DoS / DoA, eq 2 and 3
├── envs/scripted.py         §4.4 predator rule: turn at nearest prey, full throttle
├── algo/networks.py         Actor / Critic, flax
├── algo/buffer.py           flat replay buffer, one per species
├── algo/maddpg.py           make_train(cfg) -> train(key); the whole loop
├── run/train.py             config -> runs/<timestamp>_<hash8>/
├── run/replay.py            eval config -> renders/<name>/{traj.npz, config.json}
├── run/eval.py              eval config x N episodes -> evals/<name>.json
└── run/render.py            renders/<name>/ -> out.gif. no plot script yet
```

How a run is put together — the parts that are not obvious from any single file:

- **One XLA program, two nested scans.** `train(key)` is `lax.scan` over episodes,
  each of which is `lax.scan` over `episode_len` env steps. Networks, optimizer
  state, both replay buffers, env state and RNG all live in one `Carry`. Seeds are
  `vmap`ped over `make_train(cfg)` in [train.py:82](swarm_simple/run/train.py#L82).
- **`n_envs` independent envs stepped together, still one gradient step per
  species per env step.** Widening `n_envs` widens what goes into the buffer per
  step (`rows()` flattens the env axis away — the buffer has no env axis), not
  how often anyone learns. No update-every-k.
- **The critic is decentralised: `Q(o_i, a_i)`.** It is called MADDPG but there is
  no centralised critic. Deliberate, per the spec; the single most surprising fact
  in the tree.
- **No terminals.** Targets are `r + γQ'` unconditionally — episodes are
  fixed-length and a capture does not end one.
- **One buffer per species, conspecific experience merged.** `learning_starts`
  counts env steps, not transitions, because the two species insert at different
  rates.
- **`cfg` is a frozen dataclass passed to jit as a static argument.** Anything that
  changes a shape or takes a Python `if` (`n_pred`, `n_prey`, `boundary`,
  `n_neighbors`, `heading_encoding`) recompiles when changed.
- **Agents are one packed array, predators first**: `pos[:n_pred]` / `pos[n_pred:]`,
  and the same slicing on obs, actions and rewards.
- Actions are `(N, 2)` in `[-1, 1]`; `predator_prey.scale_action` owns the rescale
  into physical `(a_F, a_R)`, so the ranges live in exactly one place.
- `metrics.dos/doa` return `nan` below two conspecifics — that is why `n_pred: 0`
  is a valid config and used as a control.
- **`scripted_predator: true` swaps the learned predator for `envs/scripted.py`**
  and skips its gradient step entirely; only the prey evolve. The same rule is
  `mode: scripted` at replay time.
- **The actor loss penalises the pre-squash logits (`actor_reg`).** Without it the
  actor saturates deep in the tanh tail, the gradient dies and nothing learns —
  this was the bug, not a hyperparameter preference.
- **Training-logged `dos`/`doa` are episode means over the whole episode**, so
  they average in the formation transient from a uniform spawn and understate the
  flock. Judge flocking from `run.eval`'s `*_final_quarter`, not the training log.
- Adding a hyperparameter means editing a dataclass in `config.py`, not just the
  JSON: missing **and** unknown keys are both `SystemExit`.

Replay and rendering are two commands, and an eval config is a **simulation**, not
a picture: it carries a complete `env` of its own and says nothing about drawing.

- **An eval env owes the training env nothing but the observation width.** The actor
  is shared within a species and sees a fixed `(n, d_o)`, so a policy trained at
  3v10 replays at 3v50 in a bigger arena with different physics. Only `n_neighbors`
  and `heading_encoding` are frozen by the checkpoint — `replay.py` checks `obs_dim`
  against the run's `config.json` and exits if they disagree.
- **The model block is never written in an eval config**; each species resolves it
  from its own `run`'s `config.json`.
- Each species picks a `mode`: `learned | scripted | random | untrained`. `learned`
  and `untrained` need a `run` + `seed`; `scripted` is predators only.
- **Replay runs bare `mu(o)`** — `eps` and `noise` are a learning device, not the
  policy.
- `render.py` never simulates: it reads `traj.npz` and draws. Playback flags
  (`--fps`, `--out`) live on the CLI, never in a config.

```
configs/       training json, hand-written and COMPLETE. Missing and unknown keys are both errors.
eval_configs/  the same rules, for a replay: env_seed, pred, prey, env
runs/          append-only pool, <timestamp>_<hash8>/ with config.json, meta.json, s<seed>/{metrics.npz, params.pkl}
renders/       <name>/{traj.npz, config.json, out.gif}, one dir per eval config
evals/         <name>.json from run.eval: the eval config it came from + the numbers
```

`runs/` and `renders/` are gitignored. Both config dirs, `evals/` and
`research/notes/experiments/<exp>.md` are tracked — `evals/` is the record of
every measurement, so a result is not real until its JSON is committed.

## Layout and conventions

```
research/
├── notes/          working notes (Czech is fine here)
│   └── experiments/  one tracked <exp>.md per experiment
├── papers/         PDFs + .md transcripts — GITIGNORED except bibliography_index.md
└── code_sources/   vendored reference code, NOT our implementation
    ├── PPO_example/     JAX/gymnax single-agent PPO (supervisor's teaching repo)
    └── quad-swarm-rl/   PyTorch/Sample-Factory quadrotor swarm (upstream clone)
```

- **Papers: grep the transcript, don't open the PDF.** Every PDF has a
  `pdftotext -layout` transcript as a sibling `.md`; open the PDF only for
  figures. [bibliography_index.md](research/papers/bibliography_index.md) is the
  entry point — a tracked table of paper → files → one-line summary. Add a row
  whenever a PDF is added, and generate its transcript with
  `pdftotext -layout <file>.pdf <file>.md`.
- `research/papers/*` is gitignored (PDFs *and* transcripts); only
  `bibliography_index.md` is tracked, so it is the only record of what is there.
- **Some material under `research/papers/` is unpublished and must stay local.**
  Treat that directory as need-to-know: do not add its files to the index table,
  do not quote or summarise them in any tracked file (this one included), and
  never `git add -f` anything under it. Only add an index row for a paper that is
  publicly published. See `CLAUDE.local.md` (untracked) for anything specific.
- `quad-swarm-rl` is a **live clone with its own `.git` and an `origin` pointing at
  the upstream GitHub repo**. Never commit or push from inside it. Treat both
  `code_sources/` trees as read-only references unless the user says otherwise.

## PPO_example — JAX learner (the baseline we build on)

Has its own [CLAUDE.md](research/code_sources/PPO_example/CLAUDE.md), which is
authoritative when working inside that directory. Read it plus its `README.md`,
`ppo/train.py`, `envs/interceptor2d.py` — the whole thing is ~1,000 lines.

PureJaxRL-style PPO where the entire training loop is a single XLA program.
`ppo/train.py` (ActorCritic + `make_train`), `ppo/wrappers.py`, `ppo/config.py`
(every hyperparameter), `envs/interceptor2d.py` (env + reward + `PRESETS`),
`run/{train,eval,plot}.py` (CLI → `runs/<preset>_s<seed>/`), `cluster/` (RCI SLURM).

```bash
# PPO_example/CLAUDE.md names a conda env `agiflight`; it does NOT exist here.
# Use `conda run -n mrs-swarm` — same pins, that is where they came from.
python tests/test_smoke.py     # ~15 s, THE GATE: env checks + a learning gate
python run/train.py --steps 3e6 --num-envs 256 --num-steps 64 --num-minibatches 8   # CPU
python run/train.py            # GPU defaults: 4096 envs, 50M steps
python run/eval.py runs/straight_s0
python run/plot.py runs/straight_s0
```

Smoke tests must be green before any change is called done and before any GPU
job. Writing a new env: follow the contract in
[envs/README.md](research/code_sources/PPO_example/envs/README.md) — gymnax-style
`flax.struct` state with identical dtypes from `reset_env`/`step_env`, static
params as `pytree_node=False`, `info` carrying `terminated`/`truncated`/
`terminal_obs`, and all values chosen in `PRESETS`, never in `run/`.

The traps that repo already paid for: `env.step` auto-resets on done (log
terminal data *before* it, or use `step_env`); obs-normalization stats are part
of the policy and live in `params.pkl`; timeout is truncation (bootstrap V),
capture/miss are terminals; small batch + `ent_coef=0.01` makes entropy blow up
(keep ≥2048 envs on GPU). Cluster gotchas are in `cluster/README.md` — deploy
source by git only, verify `git log -1` on the cluster after every pull, and
`pip install --user` without `--no-deps` shadows the container's CUDA jax.

## quad-swarm-rl — PyTorch swarm reference (attention)

Separate stack, separate env: Python ≥3.11, `pip install -e .` (numpy 1.26,
torch 2.5, sample-factory, numba, gym/gymnasium). Two halves:

- `gym_art/quadrotor_multi/` — the simulator: `quadrotor_dynamics.py`,
  `quadrotor_multi.py` (the multi-agent env), `collisions/`, `obstacles/`,
  `aerodynamics/` (downwash), and `scenarios/` — one file per task
  (`static_same_goal`, `swarm_vs_swarm`, `mix`, …), each subclassing
  `scenarios/base.py`. Numba-accelerated hot paths (`--quads_use_numba`).
- `swarm_rl/` — the training glue on top of Sample Factory APPO:
  `models/quad_multi_model.py` + `models/attention_layer.py` (the neighbor
  encoder — this is the "swarming with attention" piece goal 3 points at),
  `env_wrappers/reward_shaping.py`, `runs/` (Sample Factory launcher scripts
  = experiment configs), `sim2real/` (PyTorch → C for Crazyflie firmware).

```bash
bash train.sh                                    # baseline multi-drone APPO run
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad \
    --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --quads_render=True \
    --train_dir=... --experiment=... --quads_view_mode topdown
./run_tests.sh                                   # = python -m unittest (all tests)
python -m unittest gym_art.quadrotor_multi.tests.test_multi_env   # single test module
tensorboard --logdir=./                          # from the experiment folder
```

Key training flags to recognise: `--quads_mode` (scenario), `--quads_obs_repr`,
`--quads_neighbor_encoder_type=attention`, `--quads_neighbor_visible_num`,
`--quads_use_obstacles`, `--quads_use_downwash`, `--replay_buffer_sample_prob`.
