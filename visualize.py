"""
Visualize the trained boids policy with pygame.

Usage:
    uv run python visualize.py [--checkpoint PATH] [--seed N] [--speed MULT]

Controls:
    R          – restart episode with a new random seed
    Q / Esc    – quit
"""

import argparse
import math
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pygame
from flax import nnx

import env as E
from config import EnvConfig, ModelConfig
from model import BoidPolicy
from train import load_checkpoint

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
WINDOW     = 750
PAD        = 45                           # pixel border around the world box

BG         = ( 15,  15,  25)
BORDER     = ( 70,  70, 100)
AGENT      = ( 80, 140, 220)
COLLIDING  = (220,  70,  70)
IN_GOAL    = ( 80, 210, 100)
GOAL_FILL  = ( 60, 180,  80,  45)        # RGBA – semi-transparent
GOAL_RING  = ( 60, 180,  80)
HEADING    = (220, 220, 220)
HUD_TEXT   = (180, 180, 180)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _scale(cfg: EnvConfig) -> float:
    return (WINDOW - 2 * PAD) / cfg.box_size

def world_to_px(pos, cfg: EnvConfig) -> tuple[int, int]:
    s = _scale(cfg)
    return int(PAD + pos[0] * s), int(PAD + pos[1] * s)

def world_r_to_px(r: float, cfg: EnvConfig) -> int:
    return max(1, int(r * _scale(cfg)))


# ---------------------------------------------------------------------------
# Build JIT-compiled forward pass
# ---------------------------------------------------------------------------
def build_forward(env_cfg: EnvConfig, model_cfg: ModelConfig, checkpoint: str):
    """Return (flat_params, jit_forward) where jit_forward(flat, obs) → logits."""
    dummy_key = jax.random.PRNGKey(0)
    model = BoidPolicy(env_cfg, model_cfg, rngs=nnx.Rngs(dummy_key))
    graphdef, init_state = nnx.split(model)
    _, unravel = jax.flatten_util.ravel_pytree(init_state)

    @jax.jit
    def forward(flat_params: jax.Array, obs: jax.Array) -> jax.Array:
        """obs: (N, K+1, F) → logits (N, n_actions)"""
        def single(o):
            return nnx.merge(graphdef, unravel(flat_params))(o)
        return jax.vmap(single)(obs)

    flat = load_checkpoint(checkpoint)
    # Warm up JIT
    dummy_obs = jnp.zeros((env_cfg.n_agents, *env_cfg.obs_shape))
    forward(flat, dummy_obs).block_until_ready()
    return flat, forward


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(screen, font, state: E.EnvState, cfg: EnvConfig, step: int, ep_reward: float):
    screen.fill(BG)

    agent_r = world_r_to_px(cfg.collision_radius, cfg)
    goal_r  = world_r_to_px(cfg.goal_radius, cfg)

    # --- Goal (semi-transparent circle + outline) ---
    gx, gy = world_to_px(np.array(state.goal_pos), cfg)
    fill = pygame.Surface((goal_r * 2, goal_r * 2), pygame.SRCALPHA)
    pygame.draw.circle(fill, GOAL_FILL, (goal_r, goal_r), goal_r)
    screen.blit(fill, (gx - goal_r, gy - goal_r))
    pygame.draw.circle(screen, GOAL_RING, (gx, gy), goal_r, 2)

    # --- Per-agent status ---
    pos = np.array(state.pos)                        # (N, 2)
    d2  = ((pos[:, None] - pos[None]) ** 2).sum(-1)  # (N, N)
    np.fill_diagonal(d2, np.inf)
    colliding = (d2 < cfg.collision_radius ** 2).any(1)
    in_goal   = ((pos - np.array(state.goal_pos)) ** 2).sum(1) < cfg.goal_radius ** 2

    # --- Agents ---
    for i in range(cfg.n_agents):
        ax, ay = world_to_px(pos[i], cfg)
        color  = COLLIDING if colliding[i] else (IN_GOAL if in_goal[i] else AGENT)
        pygame.draw.circle(screen, color, (ax, ay), agent_r)

        h  = float(state.heading[i])
        tip = (ax + int(agent_r * 1.8 * math.cos(h)),
               ay + int(agent_r * 1.8 * math.sin(h)))
        pygame.draw.line(screen, HEADING, (ax, ay), tip, 2)

    # --- World border ---
    box_px = WINDOW - 2 * PAD
    pygame.draw.rect(screen, BORDER, (PAD, PAD, box_px, box_px), 2)

    # --- HUD ---
    lines = [
        f"step   {step:4d} / {cfg.episode_len}",
        f"reward {ep_reward:+.3f}",
        f"in goal {int(in_goal.sum())} / {cfg.n_agents}",
        f"[R] restart  [Q] quit",
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, HUD_TEXT), (10, 10 + i * 18))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/params_latest.npy")
    parser.add_argument("--seed",  type=int,   default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    cfg       = EnvConfig()
    model_cfg = ModelConfig()

    print(f"Loading checkpoint: {args.checkpoint}")
    flat, forward = build_forward(cfg, model_cfg, args.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW, WINDOW))
    pygame.display.set_caption("Boids – Emergent Behaviour Viewer")
    font  = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    rng = jax.random.PRNGKey(args.seed)

    def new_episode():
        nonlocal rng
        rng, env_key = jax.random.split(rng)
        return E.reset(env_key, cfg), 0, 0.0   # state, step, ep_reward

    state, step, ep_reward = new_episode()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    state, step, ep_reward = new_episode()

        if step < cfg.episode_len:
            obs    = E.get_obs(state, cfg)
            logits = forward(flat, obs)
            rng, sk = jax.random.split(rng)
            actions = jax.random.categorical(sk, logits)
            state, rewards = E.step(state, actions, cfg)
            ep_reward += float(rewards.mean())
            step += 1
        else:
            # Auto-restart after a brief pause at episode end
            pygame.time.wait(1500)
            state, step, ep_reward = new_episode()

        render(screen, font, state, cfg, step, ep_reward)
        pygame.display.flip()
        clock.tick(int(cfg.fps * args.speed))


if __name__ == "__main__":
    main()
