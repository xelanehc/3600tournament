"""
net.py — JAX/Flax dual-head neural network for the carpet game.

This module implements an AlphaZero-style dual-head network:
- Input: 832 features (13 planes × 8×8 board, no rat belief)
- Trunk: Dense layers for shared representation
- Value head: Predicts board value ∈ (-1, 1)
- Policy head: Predicts move probabilities over 100 vocabulary slots

Architecture: 832 → 256 → 256 → 128 → value(64→1) + policy(64→100)

Weights are persisted as flattened JAX pytrees in .npz files for compatibility.
"""

from typing import Tuple
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import flax.linen as nn
import optax

# ── Constants ─────────────────────────────────────────────────────────────────
BOARD_SIZE  = 8
N           = BOARD_SIZE * BOARD_SIZE   # 64
N_CHANNELS  = 13
N_INPUT     = N_CHANNELS * N           # 832
N_MOVES     = 4 + 4 + 4 * 7 + N       # 100

TRUNK_DIMS  = (256, 256, 128)
VALUE_DIM   = 64
POLICY_DIM  = 64


# ── Move vocabulary helpers ───────────────────────────────────────────────────

def move_to_idx(move) -> int:
    """Convert a Move object to its vocabulary index (0–99)."""
    from game.enums import MoveType
    mt = move.move_type
    d  = int(move.direction) if move.direction is not None else 0
    if mt == MoveType.PLAIN:
        return d
    if mt == MoveType.PRIME:
        return 4 + d
    if mt == MoveType.CARPET:
        return 8 + d * 7 + (move.roll_length - 1)
    if mt == MoveType.SEARCH:
        loc = move.search_loc
        return 36 + loc[1] * BOARD_SIZE + loc[0]
    return 0

def idx_to_move(idx: int):
    """Convert vocabulary index back to a Move object."""
    from game.enums import MoveType, Direction
    from game.move import Move
    if idx < 4:
        return Move.plain(Direction(idx))
    if idx < 8:
        return Move.prime(Direction(idx - 4))
    if idx < 36:
        offset = idx - 8
        d, r = offset // 7, offset % 7 + 1
        return Move.carpet(Direction(d), r)
    cell = idx - 36
    return Move.search((cell % BOARD_SIZE, cell // BOARD_SIZE))


# ── Board → feature vector ────────────────────────────────────────────────────

def board_to_features(board) -> np.ndarray:
    """
    Encode the current board state as a (832,) float32 numpy array.
    No rat-belief channel — the network learns purely from observable board state.
    """
    from game.enums import Cell

    feat = np.zeros((N_CHANNELS, N), dtype=np.float32)
    pw   = board.player_worker
    ow   = board.opponent_worker

    def _i(loc): return loc[1] * BOARD_SIZE + loc[0]  # Cell index

    feat[0, _i(pw.get_location())] = 1.0  # Player position
    feat[1, _i(ow.get_location())] = 1.0  # Opponent position

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            i = y * BOARD_SIZE + x
            try:    c = board.get_cell((x, y))
            except: c = Cell.SPACE
            if   c == Cell.PRIMED:  feat[2, i] = 1.0
            elif c == Cell.CARPET:  feat[3, i] = 1.0
            elif c == Cell.BLOCKED: feat[4, i] = 1.0
            else:                   feat[5, i] = 1.0

    feat[6]  = float(pw.get_points())  / 100.0  # Player points
    feat[7]  = float(ow.get_points())  / 100.0  # Opponent points
    feat[8]  = float(pw.turns_left)    / 40.0   # Player turns left
    feat[9]  = float(ow.turns_left)    / 40.0   # Opponent turns left
    feat[10] = float(pw.time_left)     / 360.0  # Player time left
    feat[11] = float(ow.time_left)     / 360.0  # Opponent time left
    feat[12] = 1.0 if pw.is_player_a else 0.0   # Perspective flag

    return feat.flatten()


def legal_move_mask(board) -> np.ndarray:
    """Boolean mask (100,) of legal moves. Carpet-1 (= −1 pts) is excluded."""
    from game.enums import MoveType
    mask = np.zeros(N_MOVES, dtype=bool)
    for m in board.get_valid_moves(exclude_search=True):
        if m.move_type == MoveType.CARPET and m.roll_length == 1:
            continue
        mask[move_to_idx(m)] = True
    mask[36:100] = True   # searches always legal
    return mask


# ── Flax model ────────────────────────────────────────────────────────────────

class DualNet(nn.Module):
    """
    Dual-head network: shared trunk → value head + policy head.

    Uses @nn.compact so all Dense layers are defined inline.
    Kernel initialisation: He-uniform for trunk/head ReLU layers,
    zeros for the final bias terms (Flax default).
    """
    trunk_dims:  Tuple[int, ...] = TRUNK_DIMS
    value_dim:   int             = VALUE_DIM
    policy_dim:  int             = POLICY_DIM
    n_moves:     int             = N_MOVES

    @nn.compact
    def __call__(self, x) -> Tuple:
        """
        x : (B, N_INPUT) float32
        Returns:
            value   : (B,)         tanh-activated scalar ∈ (−1, +1)
            logits  : (B, N_MOVES) raw policy logits (pre-softmax)
        """
        # Shared trunk
        h = x
        for dim in self.trunk_dims:
            h = nn.Dense(dim, kernel_init=nn.initializers.he_uniform())(h)
            h = nn.relu(h)

        # Value head
        v = nn.Dense(self.value_dim, kernel_init=nn.initializers.he_uniform())(h)
        v = nn.relu(v)
        v = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform())(v)
        v = nn.tanh(v)
        v = v[:, 0]          # (B,)

        # Policy head
        p = nn.Dense(self.policy_dim, kernel_init=nn.initializers.he_uniform())(h)
        p = nn.relu(p)
        p = nn.Dense(self.n_moves, kernel_init=nn.initializers.glorot_uniform())(p)

        return v, p


# Module-level singleton — avoids rebuilding the Flax module on every call
_MODEL = DualNet()


# ── Parameter initialisation ──────────────────────────────────────────────────

def init_params(seed: int = 42):
    """
    Initialise network parameters using Flax's init().
    Returns the 'params' subtree of the variable collection.
    """
    key   = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, N_INPUT))
    variables = _MODEL.init(key, dummy)
    return variables['params']


# ── Optimiser factory ─────────────────────────────────────────────────────────

def make_optimizer(lr: float = 3e-4) -> optax.GradientTransformation:
    """Adam with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
        optax.adam(lr),  # Adam optimizer
    )


# ── Loss function ─────────────────────────────────────────────────────────────

def loss_fn(params, x, value_targets, policy_targets, masks):
    """
    Combined value (MSE) + policy (masked cross-entropy) loss.

    Parameters  (all JAX arrays)
    ----------
    params        : Flax params pytree
    x             : (B, N_INPUT) float32
    value_targets : (B,)         float32  ∈ [−1, 1]
    policy_targets: (B, N_MOVES) float32  one-hot or soft target
    masks         : (B, N_MOVES) bool     legal-move mask

    Returns
    -------
    total_loss : scalar
    (v_loss, p_loss) : auxiliary tuple for logging
    """
    val, logits = _MODEL.apply({'params': params}, x)

    # Value loss — MSE
    v_loss = jnp.mean((val - value_targets) ** 2)

    # Policy loss — masked log-softmax cross-entropy
    # Illegal moves get a very large negative logit so they contribute ~0 to softmax
    _NEG_INF = jnp.array(-1e9, dtype=jnp.float32)
    logits_m = jnp.where(masks, logits, _NEG_INF)
    log_sm   = logits_m - jax.nn.logsumexp(logits_m, axis=-1, keepdims=True)
    p_loss   = -jnp.mean(jnp.sum(policy_targets * log_sm, axis=-1))

    return v_loss + p_loss, (v_loss, p_loss)


# ── JIT-compiled training step ────────────────────────────────────────────────

@jit
def train_step(params, opt_state, x, value_targets, policy_targets, masks):
    """
    One JIT-compiled gradient + update step.

    The optimizer `tx` is NOT passed as an argument because jit requires
    static arguments for non-array types.  Instead, a module-level `_TX`
    singleton is used.  Call set_optimizer(lr) before training.

    Returns (new_params, new_opt_state, v_loss, p_loss).
    """
    (_, (v_loss, p_loss)), grads = value_and_grad(loss_fn, has_aux=True)(
        params, x, value_targets, policy_targets, masks)
    updates, opt_state_new = _TX.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, v_loss, p_loss


# Mutable singleton so callers can set lr before JIT compilation
_TX: optax.GradientTransformation = make_optimizer(lr=3e-4)


def set_optimizer(lr: float):
    """
    Replace the module-level optimizer singleton and clear any JIT cache.
    Call this before the first train_step() if you want a non-default lr.
    """
    global _TX, train_step
    _TX = make_optimizer(lr)
    # Re-JIT so the new _TX is captured in the closure
    train_step = jit(_train_step_raw)


def _train_step_raw(params, opt_state, x, value_targets, policy_targets, masks):
    (_, (v_loss, p_loss)), grads = value_and_grad(loss_fn, has_aux=True)(
        params, x, value_targets, policy_targets, masks)
    updates, opt_state_new = _TX.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, v_loss, p_loss


# ── JIT-compiled inference ────────────────────────────────────────────────────

@jit
def _forward_jit(params, x):
    return _MODEL.apply({'params': params}, x)


def forward(params, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference. x can be (N_INPUT,) or (B, N_INPUT) numpy float32.
    Returns (value, logits) as numpy arrays with the same leading shape.
    """
    single = x.ndim == 1
    if single:
        x = x[None]
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    val, logits = _forward_jit(params, x_jax)
    val_np    = np.asarray(val)
    logits_np = np.asarray(logits)
    if single:
        return float(val_np[0]), logits_np[0]
    return val_np, logits_np


def policy_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax → probability distribution over legal moves."""
    logits = logits.copy().astype(np.float64)
    logits[~mask] = -1e30
    logits -= logits.max()
    exp = np.exp(logits)
    exp[~mask] = 0.0
    s = exp.sum()
    if s > 0:
        return (exp / s).astype(np.float32)
    m = mask.astype(np.float32)
    return m / m.sum()


# ── Parameter serialisation ───────────────────────────────────────────────────

def save_params(params, path: str):
    """
    Flatten the Flax params pytree to numpy arrays and save as .npz.
    Keys use '/' as a separator, e.g. 'Dense_0/kernel'.
    """
    flat = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for path_tuple, leaf in leaves:
        key = '/'.join(
            p.key if hasattr(p, 'key') else str(p)
            for p in path_tuple
        )
        flat[key] = np.asarray(leaf)
    np.savez_compressed(path, **flat)
    n = sum(v.size for v in flat.values())
    print(f"  saved → {path}  ({n:,} params, {len(flat)} arrays)")


def load_params(path: str):
    """
    Load params from a .npz file and reconstruct the Flax pytree.
    Uses a reference init to recover the correct tree structure,
    then substitutes loaded leaf values.
    """
    data     = np.load(path)
    flat_loaded = {k: data[k] for k in data.files}

    # Build reference pytree to get the expected key order
    ref_params             = init_params(seed=0)
    ref_leaves, treedef   = jax.tree_util.tree_flatten_with_path(ref_params)
    ref_keys = [
        '/'.join(p.key if hasattr(p, 'key') else str(p) for p in pth)
        for pth, _ in ref_leaves
    ]

    new_leaves = [jnp.asarray(flat_loaded[k]) for k in ref_keys]
    return treedef.unflatten(new_leaves)


def param_count(params) -> int:
    """Total number of trainable parameters."""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(l.size for l in leaves)
