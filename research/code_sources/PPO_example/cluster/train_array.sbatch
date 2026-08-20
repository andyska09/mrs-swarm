#!/bin/bash
#SBATCH --job-name=ppo-ex-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/arr_%A_%a.out

# Job array: one task per (preset, seed) cell. Edit the CELLS table; the array
# index picks a row. Keep --array=0-(N-1) in sync with the table length.
#
#   sbatch cluster/train_array.sbatch            # all cells
#   sbatch --array=2 cluster/train_array.sbatch  # just cell 2
#
# %A = array job id, %a = task index — one log per cell in logs/.
set -e
cd ~/PPO_example
mkdir -p logs runs
echo "commit: $(git log --oneline -1)  task: $SLURM_ARRAY_TASK_ID  node: $(hostname)"

#            preset       seed  steps
CELLS=(     "straight     0     50e6"
            "straight     1     50e6"
            "straight     2     50e6"
            "weave        0     100e6"
            "weave        1     100e6" )

read -r PRESET SEED STEPS <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
OUT="runs/${PRESET}_s${SEED}"
echo "cell: preset=$PRESET seed=$SEED steps=$STEPS -> $OUT"

singularity exec --nv ~/containers/agifly.sif \
  python run/train.py --preset "$PRESET" --steps "$STEPS" --seed "$SEED" --out "$OUT"
singularity exec --nv ~/containers/agifly.sif \
  python run/eval.py "$OUT" --episodes 2000
