"""PPO hyperparameters. One dataclass; every value has a reason next to it.

Defaults: 4096 envs x 128 steps, 32 minibatches, 4 epochs, tanh 256x256 —
the standard PureJaxRL continuous-control setup, and what trains this env.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ── Scale ──
    num_envs: int = 4096              # parallel envs (vmap). GPU: 4096; CPU smoke: 128-256
    num_steps: int = 128              # rollout length per update
    total_timesteps: int = 50_000_000 # env steps overall (num_updates derived below)

    # ── PPO core (Schulman 2017 + the "37 details") ──
    lr: float = 3e-4
    gamma: float = 0.99               # 500-step episodes at dt=0.05 -> horizon 100 steps = 5 s. Fine here.
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Entropy bonus. CleanRL/baselines MuJoCo default to 0.0; Andrychowicz et al.
    # (2021) found no evidence it helps continuous control — Gaussian policies
    # explore through the learned log_std. 0.01 is a safe default here.
    ent_coef: float = 0.01
    # Value-loss clipping (PPO detail #9). Andrychowicz 2021: hurts. Off.
    clip_vloss: bool = False

    # ── Network ──
    actor_dims: tuple = (256, 256)
    critic_dims: tuple = ()           # () = same as actor_dims
    activation: str = "tanh"          # "tanh" | "relu"

    # ── Wrappers ──
    normalize_obs: bool = True        # Welford running mean/var. Obs SCALE is a silent killer with tanh nets.
    normalize_reward: bool = True     # divide by running std of discounted return
    clip_action: bool = True
    clip_obs: bool = False            # CleanRL clips to [-10,10]; not needed here
    clip_obs_range: float = 10.0
    clip_reward: bool = False
    clip_reward_range: float = 10.0

    # ── Truncation bootstrapping (Pardo et al. 2018) ──
    # Timeout is NOT a terminal: bootstrap gamma*V(s_T) instead of 0. Needed
    # whenever timeout != failure (here: a slow-but-closing episode). Costs one
    # extra network eval per update. The env must emit info["terminal_obs"].
    truncation_bootstrap: bool = True

    # ── LR schedule ──
    anneal_lr: bool = True

    # ── Logistics ──
    seed: int = 0

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_steps * self.num_envs)

    @property
    def minibatch_size(self) -> int:
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def effective_critic_dims(self) -> tuple:
        return self.critic_dims if self.critic_dims else self.actor_dims
