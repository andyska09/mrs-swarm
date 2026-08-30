"""Config loading. A config JSON is complete: missing and unknown keys are errors.

Fields and values follow research/notes/li2023_spec.md.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class EnvConfig:
    n_pred: int
    n_prey: int
    episode_len: int
    boundary: str  # torus | walls
    n_neighbors: int

    # Physics
    edge: float  # length (m)
    dt: float
    mass_pred: float
    mass_prey: float
    drag: float
    stiffness: float
    max_acc: float
    max_ang_vel: float
    speed_pred: float
    speed_prey: float
    perception_frac: (
        float  # R / D; D is the max possible separation, from edge and boundary
    )

    # Rewards
    catch_reward: float
    cost_af: float
    cost_ar: float
    boundary_penalty: float

    # Never stated by the paper:
    radius_pred: float
    radius_prey: float
    heading_encoding: str  # unit | angle; sets d_o to 54 or 41
    init_speed_frac: float  # spawn speed, as a fraction of that agent's max


@dataclass(frozen=True)
class ModelConfig:
    hidden: tuple
    activation: str
    actor_output: str  # squashes the actor's linear output into the action range


@dataclass(frozen=True)
class MADDPGConfig:
    episodes: int
    lr_actor: float
    lr_critic: float
    gamma: float
    tau: float
    buffer_size: int
    batch_size: int
    eps: float
    noise: float
    expl_min: float
    expl_decay: float

    optimizer: str
    learning_starts: (
        int  # env steps, not transitions: the species insert at different rates
    )

    scripted_predator: bool  # §4.4: predators follow a fixed rule, only prey learn

    # My additions:
    actor_reg: float  # weight on mean(z^2) in the actor loss


TRAIN = {"maddpg": MADDPGConfig}


MODES = ("learned", "scripted", "random", "untrained")


@dataclass(frozen=True)
class SpeciesConfig:
    mode: str
    run: str  # "" for modes that read no run
    seed: int


@dataclass(frozen=True)
class Config:
    name: str
    algo: str
    env: EnvConfig
    model: ModelConfig
    train: object


def _build(cls, blob, where):
    want = {f.name for f in fields(cls)}
    missing, unknown = want - set(blob), set(blob) - want
    if missing:
        raise SystemExit(f"{where}: missing {sorted(missing)}")
    if unknown:
        raise SystemExit(f"{where}: unknown {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in blob.items()})


def load(path):
    path = Path(path)
    blob = json.loads(path.read_text())
    want = {"name", "algo", "env", "model", "train"}
    missing, unknown = want - set(blob), set(blob) - want
    if missing or unknown:
        raise SystemExit(
            f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    if blob["algo"] not in TRAIN:
        raise SystemExit(
            f"{path}: unknown algo {blob['algo']!r}. Known: {sorted(TRAIN)}"
        )
    return Config(
        name=blob["name"],
        algo=blob["algo"],
        env=_build(EnvConfig, blob["env"], "env"),
        model=_build(ModelConfig, blob["model"], "model"),
        train=_build(TRAIN[blob["algo"]], blob["train"], "train"),
    )


def as_dict(cfg):
    return {
        "name": cfg.name,
        "algo": cfg.algo,
        "env": asdict(cfg.env),
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
    }


@dataclass(frozen=True)
class EvalConfig:
    name: str
    env_seed: int
    pred: SpeciesConfig
    prey: SpeciesConfig
    env: EnvConfig


def _check_species(sp, species, path):
    if sp.mode not in MODES:
        raise SystemExit(f"{path}: {species}.mode {sp.mode!r}. Known: {list(MODES)}")
    if sp.mode == "scripted" and species != "pred":
        raise SystemExit(f"{path}: scripted is for predators only")
    if sp.mode in ("learned", "untrained") and not sp.run:
        raise SystemExit(f"{path}: {species}.mode {sp.mode!r} needs a run")


def load_eval(path):
    path = Path(path)
    blob = json.loads(path.read_text())
    want = {"name", "env_seed", "pred", "prey", "env"}
    missing, unknown = want - set(blob), set(blob) - want
    if missing or unknown:
        raise SystemExit(
            f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    cfg = EvalConfig(
        name=blob["name"],
        env_seed=blob["env_seed"],
        pred=_build(SpeciesConfig, blob["pred"], "pred"),
        prey=_build(SpeciesConfig, blob["prey"], "prey"),
        env=_build(EnvConfig, blob["env"], "env"),
    )
    _check_species(cfg.pred, "pred", path)
    _check_species(cfg.prey, "prey", path)
    return cfg


def eval_as_dict(cfg):
    return {
        "name": cfg.name,
        "env_seed": cfg.env_seed,
        "pred": asdict(cfg.pred),
        "prey": asdict(cfg.prey),
        "env": asdict(cfg.env),
    }


def config_hash(cfg, n=8):
    payload = {k: v for k, v in as_dict(cfg).items() if k != "name"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:n]
