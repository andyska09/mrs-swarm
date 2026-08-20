#!/bin/bash
# Pull run outputs from RCI to local runs/. DATA ONLY — source code moves via
# git push/pull, never rsync (two stale-deploy incidents taught this).
#
#   ./cluster/pull_results.sh                 # everything under runs/
#   ./cluster/pull_results.sh straight_s0     # one run
RCI_USER=${RCI_USER:-pliskmic}
SUB=${1:-}
cd "$(dirname "$0")/.."
mkdir -p runs
rsync -avz --progress \
  "${RCI_USER}@login3.rci.cvut.cz:~/PPO_example/runs/${SUB}" "runs/${SUB}"
