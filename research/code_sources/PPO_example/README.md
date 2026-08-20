# PPO_example — minimal full-jit PPO + one environment

PureJaxRL-style PPO where the **entire training loop is one XLA program**, plus
the simplest useful environment: a 2D point-mass pursuer intercepting a target.
About 1,000 lines. Read it top to bottom in an afternoon; that is the point.

**Verified** (2026-08-17, Mac CPU): smoke tests green; 3 M steps in ~30 s →
capture rate 1.00; deterministic eval 28.6 ± 5.7 return, 1000/1000 captures.

```
PPO_example/
├── ppo/
│   ├── train.py          the learner: ActorCritic + make_train(config, env, env_params)
│   ├── wrappers.py       Log / ClipAction / VecEnv / NormalizeObs / NormalizeReward / TerminalObs
│   └── config.py         TrainConfig — every hyperparameter, with its reason
├── envs/
│   ├── interceptor2d.py  the environment, its reward, and its presets — one file
│   └── README.md         how to write your own env (the contract, one page)
├── run/
│   ├── train.py          CLI → runs/<name>/{params.pkl, metrics.npz, summary.json}
│   ├── eval.py           deterministic rollout of a saved policy
│   └── plot.py           training curves + trajectories → results.png
├── tests/test_smoke.py   env checks + "does it actually learn". Run before GPU.
├── cluster/              RCI/SLURM: setup, single job, job array, pull results, gotchas
└── CLAUDE.md             context for a Claude session working in this repo
```

## Quickstart

```bash
conda activate agiflight            # jax, flax, optax, distrax, gymnax — see requirements.txt
cd ~/Desktop/PPO_example
python tests/test_smoke.py          # ~15 s on CPU. All ✓ or stop here.
python run/train.py --steps 3e6 --num-envs 256 --num-steps 64 --num-minibatches 8   # CPU, ~30 s
python run/eval.py runs/straight_s0
python run/plot.py runs/straight_s0 && open runs/straight_s0/results.png
```

On a GPU drop the small-batch flags: `python run/train.py` (4096 envs, 50 M steps,
about a minute on a V100). Harder target: `--preset weave`. Cluster: `cluster/README.md`.

## The environment

Pursuer = 2D point mass, acceleration-controlled (2 actions in [-1,1] × 5 m/s²),
speed-capped at 8 m/s, light drag. Target = constant velocity (`straight`) or
evading with a sinusoidal sideways component (`weave`). Obs (7) = relative
position, relative velocity, own velocity, closing speed — **no time**.
Capture (< 0.5 m) and miss (> 20 m) end the episode; timeout (500 steps = 25 s)
is a truncation, not a failure.

Reward = `(prev_dist − dist)` progress + `−0.01‖a‖²` fuel + `+10` capture / `−10` miss.

## What is in the learner, and why

Everything defaults to what actually trained. The non-obvious choices:

- **Truncation bootstrapping** (`truncation_bootstrap=True`): timeouts are not
  failures; V at the last state is bootstrapped, not zeroed. Requires the env to
  emit `info["terminal_obs"]` — the *pre-reset* observation, because gymnax
  auto-resets and throws the real one away.
- **Obs normalization is part of the policy.** The running mean/var are saved in
  `params.pkl` and applied at eval. Forgetting this gives "trained perfectly,
  evals as random".
- **Separate actor/critic MLPs**, tanh, orthogonal init, 256×256.
- **Entropy coef 0.01, no value clipping, LR annealed** — the "37 details" and
  Andrychowicz et al. defaults.
- Metrics come out as one scalar per update (`metrics.npz`), including capture
  and miss rates read from the env's `info`.

## Rules worth keeping

1. **Test before GPU.** `tests/test_smoke.py` takes 15 s. Its learning gate
   ("return and capture rate must go up in 1 M steps") is what separates
   "compiles" from "works".
2. **Truncation ≠ termination.** Timeout must bootstrap. Get it wrong and the
   agent learns to run out the clock, or — with negative rewards — to die early.
3. **Obs scale is a silent killer.** Raw metres into a tanh net saturate units at
   init. Normalize (done here) or scale by hand.
4. **Auto-reset eats your last state.** `env.step` returns a *fresh* episode on
   `done`. Anything you log at episode end must be recorded *before* the reset.
   `run/plot.py` uses `step_env` for exactly this reason.
5. **Static config → Python `if`; traced values → `jnp.where`/`lax.cond`.**
   `evader_mode` is `pytree_node=False`, so each variant compiles its own step.
6. **State dtypes must be identical from `reset` and `step`** or `lax.scan`'s
   carry fails with an opaque error. Tested.
7. **Big batches matter.** Few hundred envs with `ent_coef=0.01` can send entropy
   up until the policy is random. On GPU keep ≥ 2048 envs; the CPU smoke config
   works only because this task is easy.
8. **Deploy code with git, move data with rsync, check the cluster hash after
   every pull.** See `cluster/README.md`.

## References

[PureJaxRL](https://github.com/luchris429/purejaxrl) ·
[CleanRL](https://github.com/vwxyzjn/cleanrl) ·
Huang et al., "The 37 Implementation Details of PPO" (2022) ·
Andrychowicz et al., "What Matters in On-Policy RL" (2021) ·
Pardo et al., "Time Limits in RL" (2018).
