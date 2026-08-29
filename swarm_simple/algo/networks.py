"""Actor and critic.  feed-forward, 3 hidden layers of 64, ReLU.

The actor emits [-1, 1]; envs.predator_prey.scale_action does the rescale into
physical (a_F, a_R), so the action ranges live in exactly one place.

Shared with any algorithm — a critic that takes (obs, action)
"""

import flax.linen as nn
import jax.numpy as jnp

ACTIVATION = {"relu": nn.relu, "tanh": nn.tanh}
SQUASH = {"tanh": nn.tanh, "sigmoid": nn.sigmoid}


class Actor(nn.Module):
    hidden: tuple
    activation: str
    squash: str
    act_dim: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for width in self.hidden:
            x = ACTIVATION[self.activation](nn.Dense(width)(x))
        return SQUASH[self.squash](nn.Dense(self.act_dim)(x))


class Critic(nn.Module):
    hidden: tuple
    activation: str

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for width in self.hidden:
            x = ACTIVATION[self.activation](nn.Dense(width)(x))
        return nn.Dense(1)(x).squeeze(-1)


def build(model_cfg, act_dim):
    """-> (actor, critic) for one species. Both are shared by its conspecifics."""
    return (
        Actor(model_cfg.hidden, model_cfg.activation, model_cfg.actor_output, act_dim),
        Critic(model_cfg.hidden, model_cfg.activation),
    )
