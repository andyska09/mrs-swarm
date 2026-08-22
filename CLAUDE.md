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

A research workspace for a thesis on multi-robot / swarm RL. Our own code lives
in `swarm/`; `research/` holds notes, papers, and two vendored reference
codebases we learn from but do not edit.

Project goals ([research/notes/goals.md](research/notes/goals.md), written in Czech):

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

## swarm/ — our code

Plan: [research/notes/plan.md](research/notes/plan.md). Replicating Li 2023 is
how the platform gets validated; the platform is the end.

**Python environments. There are two, they are not compatible, never mix them.**

| env | for | how |
|---|---|---|
| `mrs-swarm` | `swarm/` and `research/code_sources/PPO_example` | `conda run -n mrs-swarm --no-capture-output python ...` |
| — | `research/code_sources/quad-swarm-rl` | separate PyTorch stack, not created yet |

`mrs-swarm` is conda + python 3.11, CPU: jax/jaxlib 0.10.1, flax 0.12.7, optax
0.2.8, distrax 0.1.8, gymnax 0.0.9, matplotlib, imageio ([requirements.txt](requirements.txt),
pins copied from PPO_example). Use `conda run`, not `source activate`. Run
everything from the repo root as `-m swarm.…`.

```bash
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_smoke        # THE GATE, ~30 s
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_smoke --env  # env gates only, ~8 s
conda run -n mrs-swarm --no-capture-output python -m swarm.run.train --preset pendulum --exp exp_skeleton --seed 0
conda run -n mrs-swarm --no-capture-output python -m swarm.run.render --preset torus --seed 0
conda run -n mrs-swarm --no-capture-output python -m swarm.run.plot runs/exp_skeleton/pendulum/s0
```

```
swarm/
├── envs/
│   ├── predator_prey.py  env + reward + PRESETS + rollout. NOT a gymnax env: everything is per-agent
│   ├── dynamics.py       step_motion — the extension seam for 3D / drone dynamics
│   ├── metrics.py        DoS / DoA, pure functions of bare arrays
│   └── scripted.py       the paper's §4.4 predator rule; the gate-2 baseline
├── algo/
│   ├── ddpg.py           actor, critic, replay buffer, update. Species-agnostic
│   ├── train_gymnax.py   single-agent loop — the learner's regression harness
│   └── config.py         every hyperparameter, with its reason, + PRESETS
├── run/                  train.py, plot.py, render.py → runs/<exp>/<preset>/s<seed>/
└── tests/test_smoke.py
```

Using the environments:

- `swarm/envs/predator_prey.py` is a **module of functions**, not a gymnax class.
  `reset(key, params)`, `step(state, action, params)` (no key — deterministic),
  `rollout(key, params, policy)`. Agents are one packed array, predators first:
  `pos[:n_pred]` / `pos[n_pred:]`. Actions are `(N, 2)` in `[-1, 1]`; the env
  scales them (a_F is forward-only).
- `EnvParams` fields that change array shapes or take a Python `if`
  (`n_pred`, `n_prey`, `boundary`, `motion_mode`, `n_neighbors`, `episode_len`)
  are `pytree_node=False` — pass `params` as a **static** argnum when jitting,
  and changing one recompiles.
- `get_obs` returns `{self, neighbors, mask}`, **allies first**; `flatten_obs`
  produces the paper's 54-dim MLP input. An attention encoder is a second reader
  of the same structured obs, not a rewrite.
- Values are chosen in `PRESETS` (`envs/predator_prey.py` for physics,
  `algo/config.py` for training) and nowhere else. `run/` only names a preset.
- `runs/` is gitignored. `train.py` refuses a non-empty leaf without `--overwrite`.

Deviations from the paper, all deliberate, all recorded at the top of
`predator_prey.py`: agent radii (0.06 / 0.04) and initial velocity are unstated
in the paper; observations use ally-first ordering and unit-vector headings.

Smoke tests green before any change is called done.

## Layout and conventions

```
research/
├── notes/          working notes (Czech is fine here)
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
