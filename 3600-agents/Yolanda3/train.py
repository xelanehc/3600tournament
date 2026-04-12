#!/usr/bin/env python3
"""
train.py — AlphaZero-style self-play training using JAX/Flax.

This is the main training script that orchestrates the AlphaZero training loop:
self-play data collection, neural network training, and evaluation against a baseline.

Pipeline (Expert Iteration)
============================
1.  SELF-PLAY  — the current best negamax agent (agent.py, with HMM) plays
    against itself.  Every turn we record:
        state_features  : (832,) float32   board-only, no rat belief
        move_idx        : int               move vocabulary index (0–99)
        legal_mask      : (100,) bool
    At game end the outcome z ∈ {+1, 0, −1} is broadcast back to every turn.

2.  TRAIN — mini-batch gradient steps using JAX's JIT-compiled train_step:
        value  loss : MSE(v_pred, z)
        policy loss : masked cross-entropy(logits, one_hot(move_taken))

3.  EVALUATE — new params play eval_games vs the negamax baseline.
    Accept if win-rate ≥ win_threshold.

4.  ITERATE — checkpoint accepted weights; repeat.

Usage
-----
    python3 train.py                               # defaults
    python3 train.py --iterations 50 --games-per-iter 100 --resume
    python3 train.py --lr 1e-3 --batch 512 --epochs 20 --eval-games 20

Output
------
    models/weights.npz          — latest accepted params
    models/weights_iter_N.npz   — checkpoint at each accepted iteration
"""

import sys, types, importlib.util, importlib, os, argparse, random, time, pickle  # Standard library imports for system, dynamic loading, file ops, CLI, randomness, timing, serialization
from collections import deque  # For replay buffer (efficient fixed-size queue)
import numpy as np  # Numerical computations

# ── Resolve directory layout ──────────────────────────────────────────────────
# Expected layout on disk:
#   DIST/
#   ├── 3600-agents/
#   │   ├── Yolanda2/agent.py        ← baseline negamax agent
#   │   └── Yolanda3/                ← THIS folder (train.py lives here)
#   │       ├── agent_nn.py
#   │       ├── net.py
#   │       ├── train.py
#   │       └── models/
#   └── engine/
#       ├── game/                    ← game package (board.py, enums.py …)
#       └── transition_matrices/     ← bigloop.pkl, hloops.pkl …

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/3600-agents/Yolanda3
AGENTS_DIR = os.path.dirname(SCRIPT_DIR)                   # …/3600-agents
DIST_DIR   = os.path.dirname(AGENTS_DIR)                   # …/DIST
ENGINE_DIR = os.path.join(DIST_DIR, 'engine')              # …/DIST/engine
GAME_DIR   = os.path.join(ENGINE_DIR, 'game')              # …/DIST/engine/game
TM_DIR     = os.path.join(ENGINE_DIR, 'transition_matrices')

# Add engine/ to sys.path so board_utils, gameplay, etc. are importable
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

# ── Bootstrap the game package ────────────────────────────────────────────────

def _load_game():
    # Dynamically create a 'game' module and import its submodules (enums, move, worker, history, board, rat)
    # This allows importing game components without __init__.py files
    game = types.ModuleType('game')
    game.__path__ = [GAME_DIR]; game.__package__ = 'game'
    sys.modules['game'] = game
    for name in ['enums', 'move', 'worker', 'history', 'board', 'rat']:
        spec = importlib.util.spec_from_file_location(
            f'game.{name}', os.path.join(GAME_DIR, f'{name}.py'),
            submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'game'
        sys.modules[f'game.{name}'] = mod
        spec.loader.exec_module(mod)
        setattr(game, name, mod)

_load_game()

from game.enums import (
    BOARD_SIZE, MoveType, Cell, Result, WinReason, RAT_BONUS, RAT_PENALTY
)  # Game constants: board dimensions, move/cell types, win conditions, rat scoring
from game.board import Board  # Board state and game logic
from game.rat   import Rat    # Rat movement and belief tracking
from game.move  import Move   # Move representation and validation

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)  # Add this directory to path for local imports

import net as NET  # Neural network module (JAX/Flax model, training, inference)
import jax  # JAX for JIT compilation and automatic differentiation
import jax.numpy as jnp  # JAX numpy for tensor operations

# Paths to agents
AGENT_PATH    = os.path.join(AGENTS_DIR, 'Yolanda4', 'agent.py')  # Baseline negamax agent with HMM
AGENT_NN_PATH = os.path.join(SCRIPT_DIR, 'agent_nn.py')  # Neural-network guided agent


# ── Game helpers ──────────────────────────────────────────────────────────────

def _make_T():
    """
    Load a real transition matrix from engine/transition_matrices/*.pkl,
    apply up to 10% multiplicative noise (matching gameplay.py), and return
    a numpy float32 array of shape (64, 64).
    """
    pkl_files = [f for f in os.listdir(TM_DIR) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {TM_DIR!r}")
    pkl_path = os.path.join(TM_DIR, random.choice(pkl_files))  # Randomly select a transition matrix
    with open(pkl_path, 'rb') as fh:
        T = pickle.load(fh)  # Load the transition matrix (rat movement probabilities)
    T = np.array(T, dtype=np.float32)
    # Noise matching gameplay.py _load_transition_matrix()
    key   = jax.random.PRNGKey(random.randint(0, 2**32 - 1))  # Random key for noise
    noise = np.array(jax.random.uniform(key, T.shape, minval=-0.1, maxval=0.1))  # Uniform noise ±10%
    T     = np.maximum(T * (1 + noise), 0.0)  # Apply noise, ensure non-negative
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    T /= row_sum  # Normalize rows to sum to 1 (valid probability distribution)
    return T

def _set_corners(board):
    # Randomly place blocked cells in board corners to create varied starting positions
    shapes = [(2, 3), (3, 2), (2, 2)]  # Possible corner block shapes
    for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:  # Each corner
        w, h = random.choice(shapes)  # Random shape for this corner
        for dx in range(w):
            for dy in range(h):
                x = dx if ox == 0 else BOARD_SIZE - 1 - dx  # Mirror for right/bottom corners
                y = dy if oy == 0 else BOARD_SIZE - 1 - dy
                board.set_cell((x, y), Cell.BLOCKED)  # Block the cell

def _load_mod(path, name):
    # Dynamically load a Python module from file path for agent loading
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Self-play data collection ─────────────────────────────────────────────────

PLAY_TIME = 60.0  # Total time per player in seconds
INIT_BUD  = 5.0   # Initial time budget for agent initialization

def self_play_game(mod_a, mod_b, T, collect: bool = True):
    """
    Play one full game between two agent modules.

    Returns
    -------
    experiences : list of (features, move_idx, legal_mask, outcome)
                  features is (832,) float32 with NO rat-belief channel.
    a_pts, b_pts : final scores
    """
    board = Board(time_to_play=PLAY_TIME, build_history=False)  # Initialize board without history tracking
    _set_corners(board)  # Add random corner blocks
    x = random.randint(BOARD_SIZE // 2 - 2, BOARD_SIZE // 2 - 1)  # Random worker x position near center
    y = random.randint(BOARD_SIZE // 2 - 2, BOARD_SIZE // 2 + 1)  # Random worker y position
    board.player_worker.position   = (x, y)  # Set player worker position
    board.opponent_worker.position = (BOARD_SIZE - 1 - x, y)  # Mirror for opponent

    board.player_worker.time_left  = PLAY_TIME  # Set initial time
    board.opponent_worker.time_left = PLAY_TIME

    rat = Rat(T); rat.spawn()  # Initialize rat with transition matrix and spawn it

    t0 = time.perf_counter()  # Start timing for agent initialization
    try:
        agent_a = mod_a.PlayerAgent(board, T,  # Initialize agent A
                      lambda: INIT_BUD - (time.perf_counter() - t0))  # Time budget function
    except Exception as e:
        print(f"  [self-play] agent_a init failed: {e}")
        return [], 0, 0  # Return empty on failure

    t0 = time.perf_counter()
    try:
        agent_b = mod_b.PlayerAgent(board, T,
                      lambda: INIT_BUD - (time.perf_counter() - t0))
    except Exception as e:
        print(f"  [self-play] agent_b init failed: {e}")
        return [], 0, 0

    board.player_worker.time_left   = PLAY_TIME  # Reset time after initialization
    board.opponent_worker.time_left = PLAY_TIME

    trajectory = []   # List to store (features, move_idx, mask, is_player_a_turn)
    searches   = deque([(None, False), (None, False)], maxlen=2)  # Recent search outcomes

    while not board.is_game_over():  # Main game loop
        rat.move()  # Rat moves according to transition matrix
        samples   = rat.sample(board)  # Sample rat beliefs (not used by NN agent)
        is_a_turn = board.is_player_a_turn  # Whose turn it is
        agent     = agent_a if is_a_turn else agent_b  # Select agent

        # ── Collect state BEFORE the move ────────────────────────────────
        if collect:
            # Board-only features — no rat belief
            features = NET.board_to_features(board)  # Encode board state as features
            mask     = NET.legal_move_mask(board)     # Legal moves mask

        t_val   = board.player_worker.time_left  # Current time left
        t_start = time.perf_counter()  # Start timing the move
        try:
            mv = agent.play(board, samples,  # Get agent's move
                            lambda ts=t_start, tl=t_val: tl - (time.perf_counter() - ts))  # Time budget
        except Exception as e:
            board.set_winner(Result.ENEMY, WinReason.CODE_CRASH)  # Crash = opponent wins
            board.is_player_a_turn = not board.is_player_a_turn  # Switch turns
            break

        elapsed = time.perf_counter() - t_start  # Time taken for move

        if mv is None:  # Timeout
            board.set_winner(Result.ENEMY, WinReason.TIMEOUT)
            board.is_player_a_turn = not board.is_player_a_turn
            break

        if collect:
            try:   midx = NET.move_to_idx(mv)  # Convert move to vocabulary index
            except: midx = 0  # Fallback
            trajectory.append((features, midx, mask, is_a_turn))  # Store experience

        valid = board.apply_move(mv, timer=elapsed, check_ok=True)  # Apply move to board
        if not valid:
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            board.is_player_a_turn = not board.is_player_a_turn
            break
        if board.player_worker.time_left <= 0:  # Time expired
            board.set_winner(Result.ENEMY, WinReason.TIMEOUT)

        # ── Handle rat search outcome ─────────────────────────────────────
        sl = sr = None
        if mv.move_type == MoveType.SEARCH:  # If search move
            sl = mv.search_loc  # Location searched
            if mv.search_loc == rat.get_position():  # Found rat
                sr = True; rat.spawn()  # Respawn rat
                board.player_worker.increment_points(RAT_BONUS)  # Bonus points
            else:
                sr = False  # Not found
                board.player_worker.decrement_points(RAT_PENALTY)  # Penalty
        searches.append((sl, sr))  # Record search outcome

        if not board.is_game_over():
            board.reverse_perspective()  # Switch to opponent's view
            board.opponent_search = searches[-1]  # Update search history
            board.player_search   = searches[-2]

    # ── Final scores and outcomes ─────────────────────────────────────────
    if board.player_worker.is_player_a:  # Determine which worker is A/B
        a_pts = board.player_worker.get_points()
        b_pts = board.opponent_worker.get_points()
    else:
        a_pts = board.opponent_worker.get_points()
        b_pts = board.player_worker.get_points()

    if   a_pts > b_pts: z_a, z_b =  1.0, -1.0  # Win/loss outcomes
    elif b_pts > a_pts: z_a, z_b = -1.0,  1.0
    else:               z_a, z_b =  0.0,  0.0

    experiences = [
        (feat, midx, mask, z_a if is_a else z_b)  # Assign outcome to each turn
        for (feat, midx, mask, is_a) in trajectory
    ]
    return experiences, a_pts, b_pts


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int = 100_000):
        self.buf = deque(maxlen=max_size)  # Fixed-size buffer for experiences

    def add(self, experiences):
        self.buf.extend(experiences)  # Add list of experiences to buffer

    def sample(self, batch_size: int):
        batch    = random.sample(self.buf, min(batch_size, len(self.buf)))  # Random sample
        feats    = np.array([e[0] for e in batch], dtype=np.float32)  # Features batch
        midxs    = np.array([e[1] for e in batch], dtype=np.int32)    # Move indices
        masks    = np.array([e[2] for e in batch], dtype=bool)        # Legal masks
        outcomes = np.array([e[3] for e in batch], dtype=np.float32)  # Outcomes
        # One-hot policy targets
        policy_t = np.zeros((len(batch), NET.N_MOVES), dtype=np.float32)
        for i, idx in enumerate(midxs):
            if masks[i, idx]:  # If move was legal
                policy_t[i, idx] = 1.0  # One-hot for taken move
            else:
                policy_t[i] = masks[i].astype(np.float32) / max(1, masks[i].sum())  # Uniform over legal
        return feats, policy_t, masks, outcomes

    def __len__(self):
        return len(self.buf)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(params, opt_state, buf: ReplayBuffer,
                batch_size: int, n_batches: int):
    """Run n_batches gradient steps. Returns (params, opt_state, avg_v, avg_p)."""
    if len(buf) < batch_size:
        return params, opt_state, 0.0, 0.0  # Not enough data

    v_losses, p_losses = [], []  # Collect losses for averaging
    for _ in range(n_batches):
        feats, policy_t, masks, outcomes = buf.sample(batch_size)  # Sample batch
        x      = jnp.asarray(feats,    dtype=jnp.float32)  # Convert to JAX arrays
        vt     = jnp.asarray(outcomes, dtype=jnp.float32)
        pt     = jnp.asarray(policy_t, dtype=jnp.float32)
        mk     = jnp.asarray(masks,    dtype=bool)

        params, opt_state, v_loss, p_loss = NET.train_step(  # JIT training step
            params, opt_state, x, vt, pt, mk)

        v_losses.append(float(v_loss))  # Collect losses
        p_losses.append(float(p_loss))

    return params, opt_state, float(np.mean(v_losses)), float(np.mean(p_losses))  # Return updated params and avg losses


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_networks(new_params, eval_games: int, seed_offset: int) -> float:
    """
    Play eval_games games: agent_nn (new_params) vs negamax baseline.
    Returns adjusted win-rate of agent_nn.
    """
    if eval_games == 0:
        return 1.0   # auto-accept

    try:
        nn_mod = _load_mod(AGENT_NN_PATH, 'agent_nn_eval')  # Load NN agent module
    except Exception as e:
        print(f"  [eval] Could not load agent_nn: {e}")
        return 0.0

    base_mod = _load_mod(AGENT_PATH, 'agent_base_eval')  # Load baseline agent
    nn_mod._EVAL_PARAMS = new_params   # Inject new params for evaluation

    wins = ties = 0  # Counters
    for g in range(eval_games):
        np.random.seed(seed_offset + g)  # Deterministic seeding
        random.seed(seed_offset + g)
        T = _make_T()  # Random transition matrix
        if g % 2 == 0:  # Alternate first player
            _, a_pts, b_pts = self_play_game(nn_mod, base_mod, T, collect=False)
            if a_pts > b_pts:    wins += 1
            elif a_pts == b_pts: ties += 1
        else:
            _, a_pts, b_pts = self_play_game(base_mod, nn_mod, T, collect=False)
            if b_pts > a_pts:    wins += 1
            elif a_pts == b_pts: ties += 1

    win_rate = (wins + 0.5 * ties) / eval_games  # Adjusted win rate
    print(f"  [eval] {wins}W {ties}T {eval_games-wins-ties}L  win-rate={win_rate:.2f}")
    return win_rate


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero-style training with JAX/Flax')
    parser.add_argument('--iterations',     type=int,   default=20)  # Number of training iterations
    parser.add_argument('--games-per-iter', type=int,   default=50)  # Self-play games per iteration
    parser.add_argument('--epochs',         type=int,   default=10)  # Training epochs per iteration
    parser.add_argument('--batches',        type=int,   default=50)  # Gradient steps per epoch
    parser.add_argument('--batch',          type=int,   default=256) # Batch size
    parser.add_argument('--lr',             type=float, default=3e-4) # Learning rate
    parser.add_argument('--eval-games',     type=int,   default=20)  # Evaluation games per iteration
    parser.add_argument('--win-threshold',  type=float, default=0.52) # Win rate threshold to accept new network
    parser.add_argument('--weights',        type=str,   # Weights file path
                        default=os.path.join(SCRIPT_DIR, 'models', 'weights.npz'))
    parser.add_argument('--resume',         action='store_true')  # Resume from existing weights
    parser.add_argument('--buf-size',       type=int,   default=100_000) # Replay buffer size
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.weights), exist_ok=True)  # Ensure models directory exists

    # ── Set learning rate before JIT ──────────────────────────────────────
    NET.set_optimizer(args.lr)  # Configure optimizer with LR

    # ── Initialise / resume params ────────────────────────────────────────
    if args.resume and os.path.exists(args.weights):
        print(f"Resuming from {args.weights}")
        best_params = NET.load_params(args.weights)  # Load existing weights
    else:
        print("Initialising fresh weights")
        best_params = NET.init_params(seed=42)  # Initialize new weights

    n = NET.param_count(best_params)  # Count parameters
    print(f"Network: {n:,} parameters  ({NET.N_INPUT} → {NET.TRUNK_DIMS} → value+policy)")

    # ── Warm up JIT compilation ───────────────────────────────────────────
    print("Warming up JAX JIT (first call compiles)...", end=' ', flush=True)
    _tx   = NET.make_optimizer(args.lr)  # Dummy optimizer
    _os   = _tx.init(best_params)  # Dummy opt state
    dummy_x  = jnp.zeros((2, NET.N_INPUT))  # Dummy inputs
    dummy_vt = jnp.zeros((2,))
    dummy_pt = jnp.zeros((2, NET.N_MOVES))
    dummy_mk = jnp.ones((2, NET.N_MOVES), dtype=bool)
    _ = NET.train_step(best_params, _os, dummy_x, dummy_vt, dummy_pt, dummy_mk)  # Warm up JIT
    print("done")

    buf = ReplayBuffer(max_size=args.buf_size)  # Initialize replay buffer
    sp_mod = _load_mod(AGENT_PATH, 'agent_selfplay')  # Load self-play agent (baseline with HMM)

    total_games = 0  # Counter for total self-play games
    print(f"\n{'='*62}")
    print(f"Training: {args.iterations} iters × {args.games_per_iter} games/iter")
    print(f"  lr={args.lr}  batch={args.batch}  eval={args.eval_games} games")
    print(f"{'='*62}\n")

    for iteration in range(1, args.iterations + 1):  # Main training loop
        print(f"─── Iteration {iteration}/{args.iterations} ───")

        # ── Self-play ─────────────────────────────────────────────────────
        t0 = time.perf_counter()  # Start timing
        a_pts_list, b_pts_list = [], []  # Collect scores for stats
        for g in range(args.games_per_iter):
            np.random.seed(iteration * 1000 + g)  # Deterministic seeding per game
            random.seed(iteration * 1000 + g)
            T = _make_T()  # Random transition matrix
            exps, a_pts, b_pts = self_play_game(sp_mod, sp_mod, T, collect=True)  # Play game and collect experiences
            buf.add(exps)  # Add to replay buffer
            total_games += 1
            a_pts_list.append(a_pts)
            b_pts_list.append(b_pts)

        sp_time = time.perf_counter() - t0  # Self-play time
        print(f"  self-play: {args.games_per_iter} games in {sp_time:.1f}s  "
              f"| buf={len(buf):,}  "
              f"avg_pts A={np.mean(a_pts_list):.1f} B={np.mean(b_pts_list):.1f}")

        if len(buf) < args.batch:  # Not enough data
            print("  buffer too small — skipping training this iteration")
            continue

        # ── Train ─────────────────────────────────────────────────────────
        # Fresh optimizer state each iteration (matches original AlphaZero)
        candidate_params = best_params   # JAX pytrees are immutable; no copy needed
        tx      = NET.make_optimizer(args.lr)  # New optimizer
        opt_state = tx.init(candidate_params)  # Initialize opt state

        t0 = time.perf_counter()
        avg_v = avg_p = 0.0  # Accumulate losses
        for epoch in range(args.epochs):
            candidate_params, opt_state, v_l, p_l = train_epoch(  # Train for one epoch
                candidate_params, opt_state, buf, args.batch, args.batches)
            avg_v += v_l / args.epochs  # Average value loss
            avg_p += p_l / args.epochs  # Average policy loss

        train_time = time.perf_counter() - t0
        print(f"  train:     {args.epochs} epochs in {train_time:.1f}s  "
              f"| v_loss={avg_v:.4f}  p_loss={avg_p:.4f}")

        # ── Evaluate ──────────────────────────────────────────────────────
        accepted = False
        if args.eval_games > 0 and os.path.exists(AGENT_NN_PATH):  # If evaluation enabled and NN agent exists
            win_rate = evaluate_networks(  # Evaluate candidate vs baseline
                candidate_params, args.eval_games,
                seed_offset=iteration * 10_000)
            if win_rate >= args.win_threshold:  # If good enough
                best_params = candidate_params  # Accept new params
                accepted    = True
                print(f"  ✓ Accepted  (win-rate {win_rate:.2f} ≥ {args.win_threshold})")
                ckpt = args.weights.replace('.npz', f'_iter_{iteration}.npz')  # Checkpoint path
                NET.save_params(best_params, ckpt)  # Save checkpoint
            else:
                print(f"  ✗ Rejected  (win-rate {win_rate:.2f} < {args.win_threshold})")
        else:  # No evaluation or no NN agent yet
            best_params = candidate_params
            accepted    = True
            reason = "eval skipped" if args.eval_games == 0 else "no neural agent yet"
            print(f"  ✓ Accepted  ({reason})")

        NET.save_params(best_params, args.weights)  # Save latest weights
        print()

    print(f"Training complete.  Total self-play games: {total_games}")
    print(f"Final weights: {args.weights}")


if __name__ == '__main__':
    main()  # Run the training script
