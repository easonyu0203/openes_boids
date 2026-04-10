"""
Usage:
    python main.py train [--seed N]
    python main.py eval  [--checkpoint PATH] [--seed N]
"""

import argparse

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from flax import nnx

import env as boids_env
from config import EnvConfig, ESConfig, ModelConfig
from model import BoidPolicy
from train import load_checkpoint, save_checkpoint, train


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    env_cfg: EnvConfig | None = None,
    model_cfg: ModelConfig | None = None,
    seed: int = 42,
) -> None:
    env_cfg   = env_cfg   or EnvConfig()
    model_cfg = model_cfg or ModelConfig()

    # Reconstruct model structure to get graphdef + unravel_fn
    key = jax.random.PRNGKey(seed)
    key, model_key, env_key = jax.random.split(key, 3)

    model = BoidPolicy(env_cfg, model_cfg, rngs=nnx.Rngs(model_key))
    graphdef, init_state = nnx.split(model)
    _, unravel_fn = jax.flatten_util.ravel_pytree(init_state)

    flat_params = load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}  ({flat_params.shape[0]:,} params)")

    def forward_single(flat_params, obs):
        state = unravel_fn(flat_params)
        model = nnx.merge(graphdef, state)
        return model(obs)

    forward_N = jax.jit(jax.vmap(forward_single, in_axes=(None, 0)))

    state = boids_env.reset(env_key, env_cfg)
    step_rewards = []

    for t in range(env_cfg.episode_len):
        obs     = boids_env.get_obs(state, env_cfg)
        logits  = forward_N(flat_params, obs)
        key, sk = jax.random.split(key)
        actions = jax.random.categorical(sk, logits)
        state, rewards = boids_env.step(state, actions, env_cfg)
        step_rewards.append(float(rewards.mean()))

    step_rewards = np.array(step_rewards)
    print(f"Episode stats over {env_cfg.episode_len} steps:")
    print(f"  mean reward/step : {step_rewards.mean():+.4f}")
    print(f"  max  reward/step : {step_rewards.max():+.4f}")
    print(f"  min  reward/step : {step_rewards.min():+.4f}")
    print(f"  total mean reward: {step_rewards.sum():+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Boids ES experiment")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Run ES training")
    p_train.add_argument("--seed", type=int, default=0)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    p_eval.add_argument(
        "--checkpoint", type=str, default="checkpoints/params_latest.npy"
    )
    p_eval.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "train":
        train(seed=args.seed)
    elif args.cmd == "eval":
        evaluate(args.checkpoint, seed=args.seed)


if __name__ == "__main__":
    main()
