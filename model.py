"""
Lightweight transformer policy for boid agents.

Architecture:
  1. Linear projection of all K+1 tokens
  2. Add learnable ego-embedding to self token (index 0)
  3. n_blocks × EfficientAttentionBlock
       – only self token acts as Q  → O(K) attention instead of O(K²)
       – all tokens (self + neighbours) act as K, V
       – self token in all_tokens updated after each block
  4. Linear action head → 3 logits (turn-left / no-turn / turn-right)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, EnvConfig


class EfficientAttentionBlock(nnx.Module):
    """Pre-norm cross-attention where Q = self token, KV = all tokens."""

    def __init__(self, d_model: int, n_heads: int, rngs: nnx.Rngs):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj   = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj   = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        self.norm1    = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2    = nnx.LayerNorm(d_model, rngs=rngs)
        self.ffn1     = nnx.Linear(d_model, d_model * 4, rngs=rngs)
        self.ffn2     = nnx.Linear(d_model * 4, d_model, rngs=rngs)

    def __call__(
        self,
        self_tok: jax.Array,   # (d_model,)
        all_toks: jax.Array,   # (S, d_model), S = K+1
    ) -> jax.Array:            # (d_model,) updated self token
        H, Dh = self.n_heads, self.head_dim

        # Pre-norm
        n_self = self.norm1(self_tok)    # (d_model,)
        n_all  = self.norm1(all_toks)    # (S, d_model)

        # Project Q (self only), K and V (all tokens)
        q = self.q_proj(n_self)          # (d_model,)
        k = self.k_proj(n_all)           # (S, d_model)
        v = self.v_proj(n_all)           # (S, d_model)

        # Reshape for multi-head attention
        q = q.reshape(H, Dh)                             # (H, Dh)
        k = k.reshape(-1, H, Dh).transpose(1, 0, 2)     # (H, S, Dh)
        v = v.reshape(-1, H, Dh).transpose(1, 0, 2)     # (H, S, Dh)

        # Scaled dot-product attention (Q is a single vector per head)
        scale  = Dh ** -0.5
        scores = jnp.einsum("hd,hsd->hs", q, k) * scale  # (H, S)
        attn   = jax.nn.softmax(scores, axis=-1)           # (H, S)

        # Weighted sum → (H, Dh) → (d_model,)
        out = jnp.einsum("hs,hsd->hd", attn, v).reshape(self.d_model)
        out = self.out_proj(out)

        # Residual + FFN (pre-norm)
        self_tok = self_tok + out
        ffn_out  = self.ffn2(jax.nn.gelu(self.ffn1(self.norm2(self_tok))))
        self_tok = self_tok + ffn_out
        return self_tok


class BoidPolicy(nnx.Module):
    """Transformer boid policy: (K+2, feat_dim) obs → 3 action logits.

    Token layout  (must match env.get_obs):
      index 0    – self          (gets ego_emb)
      index 1..K – K neighbours  (no special embedding)
      index K+1  – goal circle   (gets goal_emb)
    """

    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig, rngs: nnx.Rngs):
        d = model_cfg.d_model
        self.input_proj  = nnx.Linear(env_cfg.feat_dim, d, rngs=rngs)
        self.ego_emb     = nnx.Param(jnp.zeros((d,)))
        self.goal_emb    = nnx.Param(jnp.zeros((d,)))
        self.blocks      = nnx.List([
            EfficientAttentionBlock(d, model_cfg.n_heads, rngs=rngs)
            for _ in range(model_cfg.n_blocks)
        ])
        self.action_head = nnx.Linear(d, model_cfg.n_actions, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        Args:
            obs: (K+2, feat_dim) — index 0 self, index K+1 goal.
        Returns:
            logits: (n_actions,)
        """
        toks = self.input_proj(obs)                      # (K+2, d_model)
        toks = toks.at[0].add(self.ego_emb.value)        # mark self token
        toks = toks.at[-1].add(self.goal_emb.value)      # mark goal token

        self_tok = toks[0]    # (d_model,)
        all_toks = toks       # (K+2, d_model), KV source

        for block in self.blocks:
            self_tok = block(self_tok, all_toks)
            # Propagate updated self into the KV context for the next block
            all_toks = all_toks.at[0].set(self_tok)

        return self.action_head(self_tok)                # (n_actions,)
