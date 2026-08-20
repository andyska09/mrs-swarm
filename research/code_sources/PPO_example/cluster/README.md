# Running on RCI (CTU cluster)

These scripts are copies of ones that run on RCI today. **Read the gotchas
before your first `sbatch`.**

## The flow

```bash
# ── local ──
git commit -am "..." && git push                # code moves via git. ALWAYS.

# ── login node (ssh pliskmic@login3.rci.cvut.cz) ──
cd ~/PPO_example && git pull
git log --oneline -1                            # <- MUST match what you pushed. See gotcha 1.
sbatch cluster/train.sbatch                     # or train_array.sbatch
squeue -u $USER
tail -f logs/train_<jobid>.out

# ── local, when done ──
./cluster/pull_results.sh                       # rsync runs/ back (data only)
python run/plot.py runs/straight_s0
```

First time only: `bash cluster/setup_rci.sh` on the login node.

## What lives where on the cluster

```
~/PPO_example/            git checkout; runs/ and logs/ are untracked
~/containers/agifly.sif   JAX + CUDA 12 container (nvcr.io/nvidia/jax:25.01-py3)
~/.local/lib/python3.12/  distrax, gymnax, tfp substrate — installed --no-deps
~/.bashrc                 export SBATCH_ACCOUNT=saskam1  SLURM_ACCOUNT=saskam1
```

Account: `pliskmic` has no personal SLURM account; jobs bill to **`saskam1`**
(Saska group) with QOS `collaborator`. The two exports in `.bashrc` make
`sbatch`/`srun` route there automatically.

If the container is missing:
```bash
mkdir -p ~/containers
singularity build ~/containers/agifly.sif docker://nvcr.io/nvidia/jax:25.01-py3   # ~10 min
```

## Partitions

| partition | GPU | queue | note |
|---|---|---|---|
| `gpu` | V100 32 GB | instant | **default here**; plenty for this env |
| `amdgpufast` | A100 40 GB | short | needs `--gres=gpu:a100:1`; "amdgpu" = AMD *host CPU*, the GPU is NVIDIA |
| `h200` / `h200fast` | H200 141 GB | small (3 nodes) | only for genuinely big runs |

For a 256×256 MLP + this tiny env, GPU tier barely matters — **batch size does**.
Going 4096 → 8192 envs bought 1.6× throughput on a similar workload; A100 vs V100
at fixed batch bought 1.04×. `--constraint=A100` does not work on RCI (no feature labels).
Interactive `srun --pty` works only in `*fast` partitions.

Expected numbers for this env: ~600k steps/s on a laptop 3080 Ti at 4096 envs;
50 M steps ≈ 80 s. A V100 will be in the same ballpark. Compile is ~10–20 s.

## Gotchas — each one cost real GPU time

1. **`git pull` can print "Updating a..b" and then abort** on an untracked-file
   collision (typically an output the cluster wrote into a path you just started
   tracking). `tail -1` of the pull hides the abort; the job then runs stale code.
   Rule: `git log --oneline -1` after every pull, compare with the pushed hash,
   *then* `sbatch`. The sbatch scripts also print the commit at the top of the log.
2. **Never `pip install` without `--no-deps` in the container.** distrax/gymnax
   pull a newer CPU-only jax into `~/.local`, which shadows the container's CUDA
   jax. Symptom: `jax.devices()` shows `CpuDevice`. Fix: `pip uninstall` the
   stray jax/jaxlib from `~/.local`, reinstall extras with `--no-deps`.
3. **Deploy source via git only.** rsync-ing individual source files has left the
   cluster checkout half-updated and killed whole job arrays. rsync is for *data*
   (the `pull_results.sh` direction).
4. **"Detected V100 not supported by this container"** from the NGC image is
   cosmetic. JAX runs.
5. **`cuda_executor driver version` warnings** are cosmetic too (driver vs
   pip-CUDA mismatch). Ignore.
6. **Small batch breaks PPO here.** With `ent_coef=0.01`, 320 envs drove entropy
   from 1.7 → 6+ (policy went maximally random). Big batches aren't just faster,
   they're algorithmically necessary at this ent_coef. Keep ≥ 2048 envs on GPU;
   the CPU smoke config (256 envs) works only because the task is trivial.
7. **Wall-clock ≠ runtime.** The first call includes XLA compile. `run/train.py`
   reports overall steps/s; for a fair sps number, run twice and time the second.

## Adapting a job script

The sbatch files take env-var overrides so you rarely edit them:

```bash
sbatch --export=ALL,PRESET=weave,STEPS=100e6,SEED=3,NUM_ENVS=8192 cluster/train.sbatch
```

For sweeps, edit the `CELLS` table in `train_array.sbatch` and keep
`--array=0-(N-1)` in sync — an off-by-one silently skips the last cell.
