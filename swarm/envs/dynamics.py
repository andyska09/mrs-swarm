"""step_motion — the extension seam.

One mode today: a 2D unicycle (Li 2023 eq 1). 3D and quadrotor dynamics become
new functions selected by the same `motion_mode`, which is a pytree_node=False
field, so the branch resolves at trace time and each mode compiles its own step.

Integration order is the paper's, and it matters:
    theta(t+1) = theta(t) + a_R dt
    v(t+1)     = v(t) + (a_F h + f_drag + f_contact) dt / m      h from the NEW theta
    x(t+1)     = x(t) + v(t) dt                                  the OLD velocity
"""
import jax.numpy as jnp


def step_motion(pos, vel, theta, action, force, params, max_speed):
    """pos (N,2) vel (N,2) theta (N,) action (N,2)=[a_F, a_R] in physical units,
    force (N,2) summed contact forces, max_speed (N,)."""
    if params.motion_mode == "unicycle2d":
        return _unicycle2d(pos, vel, theta, action, force, params, max_speed)
    raise ValueError(f"unknown motion_mode {params.motion_mode!r}")


def _unicycle2d(pos, vel, theta, action, force, params, max_speed):
    theta = wrap_angle(theta + action[:, 1] * params.dt)
    h = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    f = action[:, :1] * h - params.drag * vel + force
    new_vel = vel + f * params.dt / params.mass
    speed = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
    new_vel = new_vel * jnp.minimum(1.0, max_speed[:, None] / (speed + 1e-8))

    return pos + vel * params.dt, new_vel, theta


def wrap_angle(theta):
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
