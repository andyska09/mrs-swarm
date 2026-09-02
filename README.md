# mrs-swarm

Multi-robot / swarm reinforcement learning in JAX.

`swarm/` is a from-scratch MADDPG implementation of *"Predator–prey survival
pressure is sufficient to evolve swarming behaviors"* (Li, Li & Zhao 2023,
New J. Phys. 25 092001): predators and prey coevolve on a survival-only reward,
and swarming is expected to emerge without any swarm-specific shaping.

## Setup

Conda + Python 3.11, CPU. The environment is called `mrs-swarm`.

```bash
conda create -n mrs-swarm python=3.11
conda run -n mrs-swarm pip install -r requirements.txt
```

Everything runs from the repo root, through `conda run`:

```bash
conda run -n mrs-swarm --no-capture-output python -m swarm.run.train configs/flocking.json --seeds 0 1 2
conda run -n mrs-swarm --no-capture-output python -m swarm.run.replay eval_configs/flock50.json
conda run -n mrs-swarm --no-capture-output python -m swarm.run.render renders/flock50
conda run -n mrs-swarm --no-capture-output python -m swarm.run.eval eval_configs/flock50.json --episodes 200
```

Training is one XLA program: a scan over episodes wrapping a scan over env
steps, with seeds vmapped.

## Layout

```
swarm/
├── config.py                frozen dataclasses + load/hash — the only schema
├── envs/predator_prey.py    state, step, observe, reward, rollout
├── envs/metrics.py          degree of separation / alignment
├── envs/scripted.py         the paper's hand-written predator
├── algo/networks.py         actor and critic
├── algo/buffer.py           replay buffer, one per species
├── algo/maddpg.py           the training loop
└── run/                     train, replay, eval, render

configs/       training configs. Missing and unknown keys are both errors.
eval_configs/  the same, for a replay or an eval
runs/          <timestamp>_<hash8>/ with config.json and per-seed params
renders/       <name>/{traj.npz, out.gif}
evals/         <name>.json — the tracked record of every measurement
```

Replay and render are separate steps: an eval config describes a simulation, not
a picture, and `run/render.py` only reads `traj.npz` and draws it.

## Tests

No pytest and no aggregate gate — each module is its own runner.

```bash
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_env
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_metrics
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_networks
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_buffer
conda run -n mrs-swarm --no-capture-output python -m swarm.tests.test_maddpg
```

`CLAUDE.md` carries the details: the design decisions, the deviations from the
paper, and the experiment notes worth reading before changing anything.
