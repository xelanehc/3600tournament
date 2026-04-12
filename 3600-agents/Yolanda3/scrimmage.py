#!/usr/bin/env python3
"""
scrimmage.py — Head-to-head evaluation of the neural-network agent vs the
               baseline negamax agent (or any two agents).

This script runs multiple games between two agents, collects statistics,
and computes win rates, confidence intervals, and point differentials.
It alternates first player to remove bias and uses statistical tests
to determine significance.

Usage
-----
    # NN agent vs negamax baseline (default)
    python3 scrimmage.py

    # Custom matchup
    python3 scrimmage.py --agent-a path/to/agent_a.py --agent-b path/to/agent_b.py

    # More games for tighter confidence interval
    python3 scrimmage.py --games 50

    # Use a specific weights file
    python3 scrimmage.py --weights models/weights_iter_5.npz

Output
------
Per-game table + summary statistics:
    - Win/loss/tie counts
    - Average points for each agent
    - 95 % Wilson confidence interval on win-rate
    - Point-differential t-test
"""

import sys, types, importlib.util, importlib, os, argparse, random, time
from collections import deque
import numpy as np
import pickle

# ── Resolve directory layout ──────────────────────────────────────────────────
# Expected layout on disk:
#   DIST/
#   ├── 3600-agents/
#   │   ├── Yolanda2/agent.py        ← baseline negamax agent
#   │   └── Yolanda3/                ← THIS folder (scrimmage.py lives here)
#   └── engine/
#       ├── game/                    ← game package
#       └── transition_matrices/     ← bigloop.pkl, hloops.pkl …

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/3600-agents/Yolanda3
AGENTS_DIR = os.path.dirname(SCRIPT_DIR)                   # …/3600-agents
DIST_DIR   = os.path.dirname(AGENTS_DIR)                   # …/DIST
ENGINE_DIR = os.path.join(DIST_DIR, 'engine')
GAME_DIR   = os.path.join(ENGINE_DIR, 'game')
TM_DIR     = os.path.join(ENGINE_DIR, 'transition_matrices')

if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

# ── Bootstrap game package ────────────────────────────────────────────────────

def _load_game():
    # Dynamically load game submodules
    game = types.ModuleType('game')
    game.__path__ = [GAME_DIR]; game.__package__ = 'game'
    sys.modules['game'] = game
    for name in ['enums','move','worker','history','board','rat']:
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
    BOARD_SIZE, MAX_TURNS_PER_PLAYER, MoveType, Cell,
    Result, WinReason, RAT_BONUS, RAT_PENALTY
)  # Game constants
from game.board import Board  # Board state
from game.rat   import Rat    # Rat simulation
from game.move  import Move   # Move objects

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_A = os.path.join(SCRIPT_DIR, 'agent_nn.py')  # NN agent
DEFAULT_B = os.path.join(AGENTS_DIR, 'Yolanda2', 'agent.py')  # Baseline agent
DEFAULT_W = os.path.join(SCRIPT_DIR, 'models', 'weights.npz')  # Weights file

PLAY_TIME = 60.0  # Time per player
INIT_BUD  = 5.0   # Init budget


def _load_mod(path, name):
    # Load a Python module dynamically
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _make_T():
    # Load and noise a transition matrix
    import jax
    pkl_files = [f for f in os.listdir(TM_DIR) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {TM_DIR!r}")
    pkl_path = os.path.join(TM_DIR, random.choice(pkl_files))
    with open(pkl_path, 'rb') as fh:
        T = pickle.load(fh)
    T = np.array(T, dtype=np.float32)
    key   = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
    noise = np.array(jax.random.uniform(key, T.shape, minval=-0.1, maxval=0.1))
    T     = np.maximum(T * (1 + noise), 0.0)
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    T /= row_sum
    return T

def _set_corners(board):
    # Add random blocked corners
    shapes = [(2,3),(3,2),(2,2)]
    for ox, oy in [(0,0),(1,0),(0,1),(1,1)]:
        w, h = random.choice(shapes)
        for dx in range(w):
            for dy in range(h):
                x = dx if ox == 0 else BOARD_SIZE-1-dx
                y = dy if oy == 0 else BOARD_SIZE-1-dy
                board.set_cell((x,y), Cell.BLOCKED)


# ── Single game ────────────────────────────────────────────────────────────────
def run_game(mod_a, mod_b, T):
    # Play one game between two agents, return scores and winner
    board = Board(time_to_play=PLAY_TIME, build_history=False)
    _set_corners(board)
    x = random.randint(BOARD_SIZE//2-2, BOARD_SIZE//2-1)
    y = random.randint(BOARD_SIZE//2-2, BOARD_SIZE//2+1)
    board.player_worker.position   = (x, y)
    board.opponent_worker.position = (BOARD_SIZE-1-x, y)
    board.player_worker.time_left  = PLAY_TIME
    board.opponent_worker.time_left = PLAY_TIME

    rat = Rat(T); rat.spawn()

    t0 = time.perf_counter()
    try:
        agent_a = mod_a.PlayerAgent(board, T,
                    lambda: INIT_BUD-(time.perf_counter()-t0))
    except Exception as e:
        return None, None, 'A_INIT_FAIL', 0, str(e)

    t0 = time.perf_counter()
    try:
        agent_b = mod_b.PlayerAgent(board, T,
                    lambda: INIT_BUD-(time.perf_counter()-t0))
    except Exception as e:
        return None, None, 'B_INIT_FAIL', 0, str(e)

    board.player_worker.time_left   = PLAY_TIME
    board.opponent_worker.time_left = PLAY_TIME
    searches = deque([(None,False),(None,False)], maxlen=2)

    while not board.is_game_over():
        rat.move(); samples = rat.sample(board)
        is_a_turn = board.is_player_a_turn
        agent     = agent_a if is_a_turn else agent_b

        t_val  = board.player_worker.time_left
        t_start = time.perf_counter()
        try:
            mv = agent.play(board, samples,
                            lambda ts=t_start, tl=t_val: tl-(time.perf_counter()-ts))
        except Exception:
            board.set_winner(Result.ENEMY, WinReason.CODE_CRASH)
            board.is_player_a_turn = not board.is_player_a_turn
            break

        elapsed = time.perf_counter() - t_start
        if mv is None:
            board.set_winner(Result.ENEMY, WinReason.TIMEOUT)
            board.is_player_a_turn = not board.is_player_a_turn
            break

        valid = board.apply_move(mv, timer=elapsed, check_ok=True)
        if not valid:
            board.set_winner(Result.ENEMY, WinReason.INVALID_TURN)
            board.is_player_a_turn = not board.is_player_a_turn
            break
        if board.player_worker.time_left <= 0:
            board.set_winner(Result.ENEMY, WinReason.TIMEOUT)

        sl = sr = None
        if mv.move_type == MoveType.SEARCH:
            sl = mv.search_loc
            if mv.search_loc == rat.get_position():
                sr = True; rat.spawn()
                board.player_worker.increment_points(RAT_BONUS)
            else:
                sr = False
                board.player_worker.decrement_points(RAT_PENALTY)
        searches.append((sl, sr))
        if not board.is_game_over():
            board.reverse_perspective()
            board.opponent_search = searches[-1]
            board.player_search   = searches[-2]

    win_result = board.get_winner()
    if board.is_player_a_turn:
        winner = 'B' if win_result == Result.PLAYER else ('A' if win_result == Result.ENEMY else 'TIE')
    else:
        winner = 'A' if win_result == Result.PLAYER else ('B' if win_result == Result.ENEMY else 'TIE')

    if board.player_worker.is_player_a:
        a_pts = board.player_worker.get_points()
        b_pts = board.opponent_worker.get_points()
    else:
        a_pts = board.opponent_worker.get_points()
        b_pts = board.player_worker.get_points()

    reason = WinReason(board.win_reason).name if board.win_reason is not None else '?'
    return a_pts, b_pts, winner, board.turn_count, reason


# ── Statistics ─────────────────────────────────────────────────────────────────
def wilson_ci(wins, n, z=1.96):
    # Wilson score confidence interval for win rate
    if n == 0: return 0.0, 1.0
    p    = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre-half), min(1, centre+half)

def t_test_mean(diffs):
    # One-sample t-test for point differential
    n   = len(diffs)
    if n < 2: return float('nan'), float('nan')
    mu  = np.mean(diffs)
    se  = np.std(diffs, ddof=1) / np.sqrt(n)
    t   = mu / (se + 1e-12)
    # Approximate p-value
    from scipy.stats import t as t_dist
    p   = 2 * t_dist.sf(abs(t), df=n-1)
    return float(t), float(p)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Parse arguments, load agents, inject weights, run games, compute stats
    parser = argparse.ArgumentParser(description='Agent scrimmage')
    parser.add_argument('--agent-a',  default=DEFAULT_A,
                        help='Path to agent A module (default: agent_nn.py)')
    parser.add_argument('--agent-b',  default=DEFAULT_B,
                        help='Path to agent B module (default: negamax agent)')
    parser.add_argument('--games',    type=int, default=20,
                        help='Total games to play')
    parser.add_argument('--weights',  default=DEFAULT_W,
                        help='Weights file to inject into agent_nn')
    parser.add_argument('--seed',     type=int, default=0,
                        help='Base random seed')
    parser.add_argument('--label-a',  default=None)
    parser.add_argument('--label-b',  default=None)
    args = parser.parse_args()

    # Inject weights if agent A is agent_nn
    if os.path.basename(args.agent_a) == 'agent_nn.py' and os.path.exists(args.weights):
        if SCRIPT_DIR not in sys.path:
            sys.path.insert(0, SCRIPT_DIR)
        import net as NET
        import agent_nn
        agent_nn._EVAL_PARAMS = NET.load_params(args.weights)
        print(f"Injected weights from {args.weights}")

    print(f"Loading agents...")
    mod_a = _load_mod(args.agent_a, 'scrimmage_a')
    mod_b = _load_mod(args.agent_b, 'scrimmage_b')

    name_a = args.label_a or os.path.splitext(os.path.basename(args.agent_a))[0]
    name_b = args.label_b or os.path.splitext(os.path.basename(args.agent_b))[0]

    # Pad names to same width
    w = max(len(name_a), len(name_b), 10)
    name_a = name_a.ljust(w)
    name_b = name_b.ljust(w)

    print(f"\n{'Game':>4}  {'A='+name_a.strip():>{w+2}}  {'B='+name_b.strip():>{w+2}}  "
          f"{'Winner':>{w}}  {'Turns':>5}  Reason")
    print("─" * (w*3 + 30))

    results = []   # (winner_is_a: bool|None, a_pts, b_pts)
    a_wins = b_wins = ties = errors = 0

    for g in range(args.games):
        seed = args.seed * 10_000 + g * 137 + 42
        np.random.seed(seed); random.seed(seed)
        T = _make_T()

        # Alternate who plays A to remove first-mover bias
        if g % 2 == 0:
            play_a, play_b = mod_a, mod_b
            a_is_first = True
        else:
            play_a, play_b = mod_b, mod_a
            a_is_first = False

        t0 = time.perf_counter()
        raw_a, raw_b, winner_ab, turns, reason = run_game(play_a, play_b, T)
        elapsed = time.perf_counter() - t0

        if raw_a is None:
            print(f"{g+1:>4}  ERROR — {reason}")
            errors += 1
            continue

        # Translate raw A/B (board positions) to our agent A/B labels
        if a_is_first:
            our_a_pts, our_b_pts = raw_a, raw_b
            if winner_ab == 'A':   winner_label = name_a.strip(); a_wins += 1
            elif winner_ab == 'B': winner_label = name_b.strip(); b_wins += 1
            else:                  winner_label = 'TIE';          ties   += 1
        else:
            our_a_pts, our_b_pts = raw_b, raw_a
            if winner_ab == 'B':   winner_label = name_a.strip(); a_wins += 1
            elif winner_ab == 'A': winner_label = name_b.strip(); b_wins += 1
            else:                  winner_label = 'TIE';          ties   += 1

        results.append((our_a_pts, our_b_pts))

        # Decide which board player was "our A" for display
        if a_is_first:
            da, la = raw_a, name_a.strip()
            db, lb = raw_b, name_b.strip()
        else:
            da, la = raw_b, name_a.strip()
            db, lb = raw_a, name_b.strip()

        mark = ' ◄' if winner_label != 'TIE' else ''
        print(f"{g+1:>4}  {la}={da:>4}  {lb}={db:>4}  "
              f"{winner_label+mark:>{w+2}}  {turns:>5}  {reason}  ({elapsed:.1f}s)")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_valid = len(results)
    if n_valid == 0:
        print("\nNo valid games completed."); return

    a_pts_arr = np.array([r[0] for r in results])
    b_pts_arr = np.array([r[1] for r in results])
    diffs     = a_pts_arr - b_pts_arr

    n_games  = a_wins + b_wins + ties
    wr_a     = (a_wins + 0.5 * ties) / n_games
    ci_lo, ci_hi = wilson_ci(a_wins + 0.5*ties, n_games)
    t_stat, p_val = t_test_mean(diffs)

    print("\n" + "═" * (w*3 + 30))
    print(f"Results ({n_valid} valid games, {errors} errors):")
    print(f"  {name_a.strip():<{w}} wins: {a_wins}   avg pts: {a_pts_arr.mean():.1f}  "
          f"(σ={a_pts_arr.std():.1f})")
    print(f"  {name_b.strip():<{w}} wins: {b_wins}   avg pts: {a_pts_arr.mean():.1f}  "
          f"(σ={b_pts_arr.std():.1f})")
    print(f"  Ties: {ties}")
    print()
    print(f"  Win-rate {name_a.strip()}: {wr_a:.3f}  "
          f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Avg point differential (A−B): {diffs.mean():+.1f}  "
          f"t={t_stat:.2f}  p={p_val:.3f}")

    if p_val < 0.05:
        better = name_a.strip() if diffs.mean() > 0 else name_b.strip()
        print(f"\n  ✓ Statistically significant: {better} is better (p={p_val:.3f})")
    else:
        print(f"\n  ~ Not statistically significant yet (p={p_val:.3f}). "
              f"Run more games.")


if __name__ == '__main__':
    main()  # Run the scrimmage script
