"""Every hyperparameter, with its reason.

Values are Li 2023 table 2 unless a comment says otherwise. PRESETS is the only
place a value is chosen — run/ passes a preset name and nothing else.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ── what to train on ──
    env_id: str = "predator_prey"   # "predator_prey", or a gymnax id
    env_preset: str = "torus"       # key into envs.predator_prey.PRESETS; unused for gymnax

    # ── regime ──
    # The paper runs ONE environment: 2000 x 100 steps, one update per env step.
    # That update-to-env-step ratio decides whether off-policy learning works, so
    # parallel envs would be a different experiment.
    episodes: int = 2000
    episode_len: int = 100
    learning_starts: int = 1000     # not in the paper; buffer must hold >= batch_size to sample

    # ── networks ── 3 hidden layers x 64, ReLU, both actor and critic
    hidden_dims: tuple = (64, 64, 64)

    # ── ddpg ──
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01               # soft target update, applied every step
    buffer_size: int = 500_000
    batch_size: int = 256

    # ── exploration ── both decay per episode as x <- max(0.05, x - 5e-5)
    eps: float = 0.1                # probability of a uniform random action
    noise: float = 0.1              # gaussian std on the [-1, 1] action
    expl_min: float = 0.05
    expl_decay: float = 5e-5

    # Welford running mean/var. These stats are PART OF THE POLICY and must ship
    # with the checkpoint.
    normalize_obs: bool = True

    seed: int = 0

    @property
    def total_steps(self) -> int:
        return self.episodes * self.episode_len


PRESETS = {
    # Phase 0 — the learner's regression harness: proves the DDPG update and the
    # artifact plumbing on a known env. Pendulum's 200-step episode needs a
    # longer horizon than the paper's 0.95.
    "pendulum": TrainConfig(env_id="Pendulum-v1", episodes=300, episode_len=200,
                            gamma=0.99, buffer_size=100_000),

    # Absent neighbours are encoded as exact zeros, and obs normalisation shifts
    # them off zero into values a real neighbour could take. Hence normalize_obs=False.
    "flocking": TrainConfig(env_preset="torus", normalize_obs=False),
    "npred0": TrainConfig(env_preset="npred0", normalize_obs=False),
    "npred1": TrainConfig(env_preset="npred1", normalize_obs=False),
    "npred3": TrainConfig(env_preset="torus", normalize_obs=False),
    "rad10": TrainConfig(env_preset="torus", normalize_obs=False),
    "rad15": TrainConfig(env_preset="rad15", normalize_obs=False),
    "rad20": TrainConfig(env_preset="rad20", normalize_obs=False),
    "rad30": TrainConfig(env_preset="rad30", normalize_obs=False),
    "speed11": TrainConfig(env_preset="torus", normalize_obs=False),
    "speed53": TrainConfig(env_preset="speed_53", normalize_obs=False),
    "speed35": TrainConfig(env_preset="speed_35", normalize_obs=False),
    "perc33": TrainConfig(env_preset="torus", normalize_obs=False),
    "perc23": TrainConfig(env_preset="perception_23", normalize_obs=False),
    "perc13": TrainConfig(env_preset="perception_13", normalize_obs=False),
}


def get_train_config(preset: str) -> TrainConfig:
    assert preset in PRESETS, f"Unknown preset {preset!r}. Available: {list(PRESETS)}"
    return PRESETS[preset]
