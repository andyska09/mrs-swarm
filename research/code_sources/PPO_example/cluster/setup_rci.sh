#!/bin/bash
# One-time RCI setup for PPO_example. Run ON THE LOGIN NODE (login3.rci.cvut.cz).
#
# Assumes the shared JAX+CUDA container already exists at ~/containers/agifly.sif
# (built once from nvcr.io/nvidia/jax:25.01-py3 — see cluster/README.md if not).
# The container supplies jax/jaxlib(CUDA)/flax/optax; distrax + gymnax live in
# ~/.local via `pip install --user --no-deps` (NEVER without --no-deps: it pulls a
# CPU-only newer jax into ~/.local that shadows the container's CUDA jax).
set -e

REPO=${REPO:-git@github.com:majky1997-arch/PPO_example.git}   # adjust if hosted elsewhere
BRANCH=${BRANCH:-main}

if [ ! -d ~/PPO_example ]; then
  git clone "$REPO" ~/PPO_example
fi
cd ~/PPO_example
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
echo "checked out: $(git log --oneline -1)"      # compare with what you pushed!

# Extras the container does not ship. --no-deps is load-bearing.
singularity exec ~/containers/agifly.sif pip install --user --no-deps distrax gymnax
# distrax needs tensorflow-probability's substrate (may already be present; harmless to repeat):
singularity exec ~/containers/agifly.sif pip install --user --no-deps \
    tfp-nightly dm-tree cloudpickle gast attrs wrapt 2>/dev/null || true

# Verify: CUDA jax must win, and our imports must resolve.
singularity exec --nv ~/containers/agifly.sif python - <<'EOF'
import jax, jaxlib, distrax, gymnax
print("jax", jax.__version__, "| jaxlib", jaxlib.__version__, "| devices:", jax.devices())
import sys; sys.path.insert(0, ".")
from envs import Interceptor2D, get_env_params
from ppo.train import make_train
print("imports ok; presets:", list(__import__('envs').PRESETS))
EOF
mkdir -p logs runs
echo "Setup OK. Next: sbatch cluster/train.sbatch"
