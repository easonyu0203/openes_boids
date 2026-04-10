"""
OpenAI ES training for the boid policy.

Key design choices:
  • All N agents share one policy (parameter sharing).
  • Flat-param approach: params ↔ 1-D JAX array via jax.flatten_util.ravel_pytree.
    This makes noise perturbation trivial and avoids pytree vmapping complexity.
  • Antithetic (mirror) sampling: each noise draw ε is evaluated as +ε and -ε.
  • Pre-allocated noise table to avoid repeated RNG calls inside the JIT boundary.
  • Same env seed per generation so fitness differences reflect policy quality,
    not environment randomness. Each population member gets its own action key.
  • Rank-normalised fitness shaping for variance reduction.
  • Adam optimiser (via optax) applied to the ES gradient estimate.
  • Checkpoints saved as .npy files; training stats logged to CSV.
"""

import csv
import pathlib
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import env as boids_env
from config import EnvConfig, ESConfig, ModelConfig
from model import BoidPolicy


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_rollout(
    env_cfg: EnvConfig,
    graphdef,
    unravel_fn,
) -> callable:
    """
    Return a JIT+vmap-compiled function:
        rollout_pop(pop_flat, env_key, action_keys) → fitnesses (pop_size,)

    pop_flat    : (pop_size, D)  – perturbed flat params for each member
    env_key     : ()             – single key for env reset (shared)
    action_keys : (pop_size, 2)  – per-member keys for action sampling
    """

    def forward_single(flat_params: jax.Array, obs: jax.Array) -> jax.Array:
        """Single-agent forward pass. obs: (K+1, feat_dim) → logits (n_actions,)"""
        state = unravel_fn(flat_params)
        model = nnx.merge(graphdef, state)
        return model(obs)

    # vmap over N agents (same params, each agent has its own obs slice)
    forward_N = jax.vmap(forward_single, in_axes=(None, 0))

    def rollout_one(flat_params: jax.Array, env_key: jax.Array, action_key: jax.Array) -> jax.Array:
        """Run one episode. Returns mean per-agent reward per step."""
        state = boids_env.reset(env_key, env_cfg)

        def scan_step(carry, _):
            env_state, cum_reward, rng = carry
            obs = boids_env.get_obs(env_state, env_cfg)         # (N, K+1, 11)
            logits = forward_N(flat_params, obs)                 # (N, 3)
            rng, subkey = jax.random.split(rng)
            actions = jax.random.categorical(subkey, logits)    # (N,)
            env_state, rewards = boids_env.step(env_state, actions, env_cfg)
            return (env_state, cum_reward + rewards.mean(), rng), None

        (_, total, _), _ = jax.lax.scan(
            scan_step,
            (state, jnp.float32(0.0), action_key),
            None,
            length=env_cfg.episode_len,
        )
        return total / env_cfg.episode_len

    # vmap over population: params vary, env_key fixed, action_key varies
    rollout_pop = jax.jit(jax.vmap(rollout_one, in_axes=(0, None, 0)))
    return rollout_pop


# ---------------------------------------------------------------------------
# ES utilities
# ---------------------------------------------------------------------------

def rank_normalize(fitnesses: jax.Array) -> jax.Array:
    """Map fitnesses to centred uniform ranks in [-0.5, 0.5]."""
    n = fitnesses.shape[0]
    ranks = jnp.argsort(jnp.argsort(fitnesses))   # 0-indexed integer ranks
    return ranks.astype(jnp.float32) / (n - 1) - 0.5


def es_grad(fitnesses: jax.Array, eps: jax.Array, noise_std: float) -> jax.Array:
    """
    Antithetic ES gradient estimate.
      fitnesses : (pop_size,)  – first half = +ε evaluations
      eps       : (half, D)   – noise draws (positive half only)
    Returns gradient (D,) for maximisation.
    """
    half = eps.shape[0]
    F_pos = fitnesses[:half]         # (half,)
    F_neg = fitnesses[half:]         # (half,)

    # Fitness shaping over all 2*half evaluations
    all_fits = jnp.concatenate([F_pos, F_neg])
    shaped   = rank_normalize(all_fits)
    F_pos_s  = shaped[:half]
    F_neg_s  = shaped[half:]

    grad = jnp.mean((F_pos_s - F_neg_s)[:, None] * eps, axis=0) / noise_std
    return grad


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(flat_params: jax.Array, gen: int, cfg: ESConfig) -> None:
    directory = pathlib.Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    arr = np.array(flat_params)
    np.save(directory / f"params_gen{gen:06d}.npy", arr)
    np.save(directory / "params_latest.npy", arr)


def load_checkpoint(path: str) -> jax.Array:
    return jnp.array(np.load(path))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    env_cfg: EnvConfig | None = None,
    model_cfg: ModelConfig | None = None,
    es_cfg: ESConfig | None = None,
    seed: int = 0,
) -> jax.Array:
    """Run ES training. Returns final flat params."""
    env_cfg   = env_cfg   or EnvConfig()
    model_cfg = model_cfg or ModelConfig()
    es_cfg    = es_cfg    or ESConfig()

    key = jax.random.PRNGKey(seed)

    # --- Initialise model ---
    key, model_key = jax.random.split(key)
    model = BoidPolicy(env_cfg, model_cfg, rngs=nnx.Rngs(model_key))
    graphdef, init_state = nnx.split(model)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(init_state)
    D = flat_params.shape[0]
    print(f"Policy parameter count: {D:,}")

    # --- Pre-allocate noise table ---
    key, noise_key = jax.random.split(key)
    noise_table = jax.random.normal(noise_key, (es_cfg.noise_table_size, D))

    # --- Build rollout ---
    rollout_pop = build_rollout(env_cfg, graphdef, unravel_fn)

    # Warm-up JIT (small run to avoid counting compile time)
    print("Compiling rollout (first call)...")
    half = es_cfg.pop_size // 2
    _dummy_params = jnp.tile(flat_params[None, :], (es_cfg.pop_size, 1))
    _dummy_env_k  = jax.random.PRNGKey(0)
    _dummy_act_ks = jax.random.split(jax.random.PRNGKey(1), es_cfg.pop_size)
    rollout_pop(_dummy_params, _dummy_env_k, _dummy_act_ks).block_until_ready()
    print("Done.")

    # --- Optimizer ---
    optimizer  = optax.adam(es_cfg.lr)
    opt_state  = optimizer.init(flat_params)

    # --- Logging ---
    log_path = pathlib.Path(es_cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["gen", "mean_fitness", "max_fitness", "min_fitness", "elapsed_s"])

    t0 = time.time()

    for gen in range(es_cfg.n_generations):
        key, env_key, noise_key, action_key = jax.random.split(key, 4)

        # Antithetic noise sampling from pre-allocated table
        noise_idx = jax.random.randint(noise_key, (half,), 0, es_cfg.noise_table_size)
        eps = noise_table[noise_idx]                           # (half, D)

        pos_params = flat_params[None, :] + es_cfg.noise_std * eps   # (half, D)
        neg_params = flat_params[None, :] - es_cfg.noise_std * eps   # (half, D)
        all_params = jnp.concatenate([pos_params, neg_params], axis=0)  # (pop_size, D)

        # Per-member action keys (env dynamics identical across population)
        action_keys = jax.random.split(action_key, es_cfg.pop_size)

        fitnesses = rollout_pop(all_params, env_key, action_keys)    # (pop_size,)

        # ES gradient and optimizer step (negate for maximisation)
        grad    = es_grad(fitnesses, eps, es_cfg.noise_std)
        updates, opt_state = optimizer.update(-grad, opt_state)
        flat_params = optax.apply_updates(flat_params, updates)

        if gen % es_cfg.log_every == 0:
            mean_f = float(fitnesses.mean())
            max_f  = float(fitnesses.max())
            min_f  = float(fitnesses.min())
            elapsed = time.time() - t0
            print(
                f"gen {gen:5d} | mean={mean_f:+.4f}  max={max_f:+.4f}"
                f"  min={min_f:+.4f} | {elapsed:.1f}s"
            )
            writer.writerow([gen, mean_f, max_f, min_f, f"{elapsed:.1f}"])
            csv_file.flush()

        if gen % es_cfg.checkpoint_every == 0:
            save_checkpoint(flat_params, gen, es_cfg)

    # Final checkpoint
    save_checkpoint(flat_params, es_cfg.n_generations, es_cfg)
    csv_file.close()
    print("Training complete.")
    return flat_params
