"""Create a run from a config JSON.

python -m swarm_simple.run.train configs/flocking.json --seeds 0 1 2
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from swarm_simple.config import as_dict, config_hash, load

ROOT = Path(__file__).resolve().parents[2]


def commit():
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def make_run(cfg, seeds):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = config_hash(cfg)
    run = ROOT / "runs" / f"{stamp}_{digest}"
    for s in seeds:
        (run / f"s{s}").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(as_dict(cfg), indent=2))
    (run / "meta.json").write_text(
        json.dumps(
            {
                "created": stamp,
                "config_hash": digest,
                "commit": commit(),
                "seeds": list(seeds),
            },
            indent=2,
        )
    )
    return run


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    cfg = load(a.config)
    run = make_run(cfg, a.seeds)
    print(f"{cfg.name}  {cfg.algo}  seeds={a.seeds}")
    print(f"-> {run.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
