# CLAUDE.md — PPO_example

## What this is

A minimal, verified, standalone JAX PPO + one environment (2D interception).
Purpose: a clean starting point for students building their own environments,
and a reference for how we do PPO. It is a teaching repo — keep it small. Do
not add features nobody asked for.

**Owner:** Michal Pliska, PhD student, CTU Prague (FEL). Blunt-feedback culture;
if a number smells, trace one example end-to-end before defending the machinery.

## Read first

`README.md` (quickstart + the rules), then `ppo/train.py` top to bottom, then
`envs/interceptor2d.py`. `cluster/README.md` before touching RCI. The whole
thing is ~1,000 lines; read it.

## Contracts

- **Smoke tests are the gate.** `python tests/test_smoke.py` must be green
  before any change is called done and before any GPU job. It includes a
  learning gate (return + capture rate must rise in 1 M CPU steps). If you touch
  the env or reward, add an assertion for what you changed.
- **Verify, don't assert.** After a change: run the smoke test, run
  `run/train.py` at CPU scale, run `run/eval.py` and check the eval numbers match
  the training tail (today: 28.3 vs 28.6). Report actual output.
- **Values are chosen in `envs/*.py` PRESETS and `ppo/config.py` only.** No
  hard-coded numbers in `run/`.

## Environment

```bash
conda activate agiflight        # arm64 Mac; CPU JAX. jax 0.10, flax 0.12, optax, distrax, gymnax
# `conda run` is broken in sandboxed shells — call the binary directly:
/opt/homebrew/Caskroom/miniforge/base/envs/agiflight/bin/python tests/test_smoke.py
```

Cluster: RCI, user `pliskmic`, account `saskam1`, container `~/containers/agifly.sif`.
Deploy = `git push` + `git pull` on the cluster; **verify `git log -1` after the
pull** before `sbatch`; rsync is for results only. Ask before pushing or
submitting anything.

## Style

- Blunt, minimal code. No base classes, protocols, registries. One file per
  concern, readable top to bottom.
- Comments state constraints and reasons, not history.
- flax.struct pytrees for all state; static config → `pytree_node=False` →
  Python `if` at trace time; traced values → `jnp.where`/`lax.cond`. Fixed
  shapes everywhere. Same dtypes from reset and step.

## Traps this codebase already knows about

- `env.step` auto-resets on done — record anything terminal *before* it, or use
  `step_env`. Every trajectory ending at (0,0) means you forgot.
- Obs-normalization stats are part of the policy; they're in `params.pkl`.
- Timeout is truncation (bootstrap V), capture/miss are terminals (V → 0).
- Small batch + `ent_coef=0.01` → entropy blows up. Keep ≥ 2048 envs on GPU.
- Cluster: `pip install --user` **without** `--no-deps` shadows CUDA jax; and
  `git pull` can silently abort on an untracked-file collision.
