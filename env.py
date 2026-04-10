"""
JAX boids environment.

State layout (all arrays, pure-functional):
  pos      (N, 2)  – agent positions in [0, box_size]²
  heading  (N,)    – agent heading in radians (world frame)
  omega    (N,)    – current angular velocity (last applied turn)
  goal_pos (2,)    – center of the moving food circle
  goal_vel (2,)    – velocity of the food circle
  step     ()      – integer step counter

Observation per agent: (K+1, 11), token 0 = self.
Feature layout per token:
  [vx, vy, omega, d_left, d_right, d_bottom, d_top,
   dist_to_goal, angle_to_goal,   ← ego-centric angle
   dist_to_ego,  angle_to_ego]    ← ego-centric angle; both 0 for self token
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from config import EnvConfig


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    pos: jax.Array       # (N, 2)
    heading: jax.Array   # (N,)
    omega: jax.Array     # (N,)
    goal_pos: jax.Array  # (2,)
    goal_vel: jax.Array  # (2,)
    step: jax.Array      # scalar int


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset(key: jax.Array, cfg: EnvConfig) -> EnvState:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    margin = cfg.collision_radius * 2
    pos = jax.random.uniform(
        k1, (cfg.n_agents, 2),
        minval=margin, maxval=cfg.box_size - margin,
    )
    heading = jax.random.uniform(k2, (cfg.n_agents,), minval=-jnp.pi, maxval=jnp.pi)
    omega = jnp.zeros(cfg.n_agents)

    goal_margin = cfg.goal_radius
    goal_pos = jax.random.uniform(
        k3, (2,),
        minval=goal_margin, maxval=cfg.box_size - goal_margin,
    )
    # Random initial direction at fixed speed
    angle = jax.random.uniform(k4, (), minval=-jnp.pi, maxval=jnp.pi)
    goal_vel = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * cfg.goal_speed

    return EnvState(pos=pos, heading=heading, omega=omega,
                    goal_pos=goal_pos, goal_vel=goal_vel,
                    step=jnp.int32(0))


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def step(
    state: EnvState,
    actions: jax.Array,   # (N,) int in {0=left, 1=none, 2=right}
    cfg: EnvConfig,
) -> tuple[EnvState, jax.Array]:
    """Advance simulation by one timestep. Returns (new_state, rewards)."""
    dt = cfg.dt

    # --- Agent physics ---
    # Map action {0,1,2} → turn ∈ {-1, 0, +1} scaled by turn_rate
    turn = (actions.astype(jnp.float32) - 1.0) * cfg.turn_rate  # (N,)
    new_heading = state.heading + turn * dt
    # Normalise to (-π, π]
    new_heading = (new_heading + jnp.pi) % (2 * jnp.pi) - jnp.pi

    vx = jnp.cos(new_heading) * cfg.agent_speed  # (N,)
    vy = jnp.sin(new_heading) * cfg.agent_speed  # (N,)
    new_pos = state.pos + jnp.stack([vx, vy], axis=-1) * dt  # (N, 2)

    # Wall bounce: reflect velocity component, clamp position
    hit_left   = new_pos[:, 0] < 0.0
    hit_right  = new_pos[:, 0] > cfg.box_size
    hit_bottom = new_pos[:, 1] < 0.0
    hit_top    = new_pos[:, 1] > cfg.box_size

    new_vx = jnp.where(hit_left | hit_right,  -vx, vx)
    new_vy = jnp.where(hit_bottom | hit_top,  -vy, vy)
    new_heading = jnp.arctan2(new_vy, new_vx)

    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], 0.0, cfg.box_size),
        jnp.clip(new_pos[:, 1], 0.0, cfg.box_size),
    ], axis=-1)

    # --- Goal physics (billiard random walk) ---
    gm = cfg.goal_radius
    new_goal_pos = state.goal_pos + state.goal_vel * dt
    g_hit_h = (new_goal_pos[0] < gm) | (new_goal_pos[0] > cfg.box_size - gm)
    g_hit_v = (new_goal_pos[1] < gm) | (new_goal_pos[1] > cfg.box_size - gm)
    new_goal_vel = jnp.where(
        jnp.array([g_hit_h, g_hit_v]),
        -state.goal_vel,
        state.goal_vel,
    )
    new_goal_pos = jnp.clip(new_goal_pos, gm, cfg.box_size - gm)

    # --- Rewards ---
    hit_wall = hit_left | hit_right | hit_bottom | hit_top
    wall_rew = jnp.where(hit_wall, cfg.wall_penalty, 0.0)

    # Pairwise distances for collision detection
    diff = new_pos[:, None, :] - new_pos[None, :, :]   # (N, N, 2)
    dist2 = (diff ** 2).sum(-1)                          # (N, N)
    no_self = ~jnp.eye(cfg.n_agents, dtype=bool)
    colliding = (dist2 < cfg.collision_radius ** 2) & no_self  # (N, N)
    collision_rew = jnp.where(colliding.any(axis=1), cfg.collision_penalty, 0.0)

    goal_dist2 = ((new_pos - new_goal_pos) ** 2).sum(-1)  # (N,)
    goal_rew = jnp.where(goal_dist2 < cfg.goal_radius ** 2, cfg.goal_reward, 0.0)

    rewards = wall_rew + collision_rew + goal_rew  # (N,)

    new_state = EnvState(
        pos=new_pos,
        heading=new_heading,
        omega=turn,
        goal_pos=new_goal_pos,
        goal_vel=new_goal_vel,
        step=state.step + 1,
    )
    return new_state, rewards


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def get_obs(state: EnvState, cfg: EnvConfig) -> jax.Array:
    """
    Build observation tensor of shape (N, K+1, 11).
    Token 0 is the self token; tokens 1..K are the K closest other agents.
    If n_agents-1 < K the remaining slots are zero-padded.
    """
    N = cfg.n_agents
    K = cfg.n_obs_agents
    actual_k = min(K, N - 1)      # how many real neighbours we can supply

    # --- Per-agent absolute features (world frame) ---
    vx = jnp.cos(state.heading) * cfg.agent_speed   # (N,)
    vy = jnp.sin(state.heading) * cfg.agent_speed   # (N,)
    omega = state.omega                               # (N,)

    d_left   = state.pos[:, 0]
    d_right  = cfg.box_size - state.pos[:, 0]
    d_bottom = state.pos[:, 1]
    d_top    = cfg.box_size - state.pos[:, 1]

    # Ego-centric angle to goal for each agent
    goal_diff = state.goal_pos[None, :] - state.pos   # (N, 2)
    dist_to_goal = jnp.sqrt((goal_diff ** 2).sum(-1) + 1e-8)  # (N,)
    angle_to_goal_world = jnp.arctan2(goal_diff[:, 1], goal_diff[:, 0])
    angle_to_goal = _wrap(angle_to_goal_world - state.heading)  # (N,)

    # Stack absolute+goal features: (N, 9)
    abs_feats = jnp.stack(
        [vx, vy, omega, d_left, d_right, d_bottom, d_top, dist_to_goal, angle_to_goal],
        axis=-1,
    )

    # --- Pairwise relative features ---
    diff = state.pos[None, :, :] - state.pos[:, None, :]   # (N, N, 2)
    #   diff[i, j] = pos[j] - pos[i]  → vector FROM i TO j
    pairwise_dist = jnp.sqrt((diff ** 2).sum(-1) + 1e-8)   # (N, N)

    angle_world = jnp.arctan2(diff[:, :, 1], diff[:, :, 0])       # (N, N)
    # ego-centric: subtract observer heading
    angle_ego = _wrap(angle_world - state.heading[:, None])        # (N, N)

    # Mask self with large distance so it sorts last
    self_mask = jnp.eye(N, dtype=jnp.float32) * 1e9
    masked_dist = pairwise_dist + self_mask                        # (N, N)

    # Indices of actual_k nearest neighbours for each agent
    nearest_idx = jnp.argsort(masked_dist, axis=1)[:, :actual_k]  # (N, actual_k)

    # Gather absolute features for neighbours: (N, actual_k, 9)
    other_abs = abs_feats[nearest_idx]

    # Gather relative dist/angle: (N, actual_k)
    row_idx = jnp.arange(N)[:, None]
    other_dist  = pairwise_dist[row_idx, nearest_idx]
    other_angle = angle_ego[row_idx, nearest_idx]

    # Concatenate to (N, actual_k, 11)
    other_feats = jnp.concatenate(
        [other_abs, other_dist[:, :, None], other_angle[:, :, None]],
        axis=-1,
    )

    # Zero-pad to (N, K, 11) if needed
    if actual_k < K:
        pad = jnp.zeros((N, K - actual_k, cfg.feat_dim))
        other_feats = jnp.concatenate([other_feats, pad], axis=1)

    # --- Self token ---
    self_feats = jnp.concatenate(
        [abs_feats, jnp.zeros((N, 2))],   # dist_to_ego=0, angle_to_ego=0
        axis=-1,
    )  # (N, 11)

    # Stack: self first, then neighbours → (N, K+1, 11)
    obs = jnp.concatenate([self_feats[:, None, :], other_feats], axis=1)
    return obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(angle: jax.Array) -> jax.Array:
    """Wrap angle to (-π, π]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
