#!/bin/bash
#SBATCH --job-name=ppo-ex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out

# Single training run. Account comes from SBATCH_ACCOUNT=saskam1 in ~/.bashrc.
#
# Partition cheat-sheet (RCI, 2026):
#   gpu          V100 32 GB   instant queue        <- default; fine for this env
#   amdgpufast   A100 40 GB   (use --gres=gpu:a100:1; "amdgpu" = AMD host CPU, NOT AMD GPU)
#   h200         H200 141 GB  small queue, only for genuinely big runs
# Interactive `srun --pty` only works in *fast partitions.
#
# Override anything at submit time, e.g.:
#   sbatch --export=ALL,PRESET=weave,STEPS=100e6,SEED=3 cluster/train.sbatch
set -e
cd ~/PPO_example
mkdir -p logs runs
echo "commit: $(git log --oneline -1)"          # the stale-checkout guard: eyeball this in the log
echo "node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES"

PRESET=${PRESET:-straight}
STEPS=${STEPS:-50e6}
SEED=${SEED:-0}
NUM_ENVS=${NUM_ENVS:-4096}

singularity exec --nv ~/containers/agifly.sif \
  python run/train.py --preset "$PRESET" --steps "$STEPS" --seed "$SEED" \
    --num-envs "$NUM_ENVS" --out "runs/${PRESET}_s${SEED}"

singularity exec --nv ~/containers/agifly.sif \
  python run/eval.py "runs/${PRESET}_s${SEED}" --episodes 2000
