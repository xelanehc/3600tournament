import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    RAT_BONUS, RAT_PENALTY, CARPET_POINTS_TABLE,
    loc_after_direction, MAX_TURNS_PER_PLAYER,
)
from game.move import Move
from game.board import Board

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants & Global Engine Tools
# ═══════════════════════════════════════════════════════════════════════════════

N = BOARD_SIZE * BOARD_SIZE
ALL_DIRS = list(Direction)
INF = float('inf')

# TT Flags
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2

class Zobrist:
    """Provides unique 64-bit hashes for board states to avoid re-calculating work."""
    def __init__(self):
        self.table = np.random.randint(0, 2**63, (BOARD_SIZE, BOARD_SIZE, 5), dtype=np.uint64)
        self.turn = np.random.randint(0, 2**63, dtype=np.uint64)
        self.workers = np.random.randint(0, 2**63, (2, BOARD_SIZE, BOARD_SIZE), dtype=np.uint64)

    def get_hash(self, board: Board) -> int:
        h = np.uint64(0)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell = board.get_cell((x, y))
                h ^= self.table[x][y][cell.value]
        
        p_loc = board.player_worker.get_location()
        o_loc = board.opponent_worker.get_location()
        h ^= self.workers[0][p_loc[0]][p_loc[1]]
        h ^= self.workers[1][o_loc[0]][o_loc[1]]
        
        if board.turn_count % 2 == 1:
            h ^= self.turn
        return int(h)

ZOBRIST = Zobrist()

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM Belief Tracking
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.30, 0.20),
    Cell.SPACE:   (0.70, 0.15, 0.15),
    Cell.PRIMED:  (0.10, 0.80, 0.10),
    Cell.CARPET:  (0.10, 0.10, 0.80),
}
DIST_ERR_P = (0.12, 0.70, 0.12, 0.06)
DIST_ERR_OFF = (-1, 0, 1, 2)

class RatBelief:
    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.b = np.full(N, 1.0 / N)

    def predict(self):
        self.b = self.b @ self.T
        self._norm()

    def update_noise(self, board: Board, noise_type):
        ni = int(noise_type)
        for i in range(N):
            cell = board.get_cell((i % BOARD_SIZE, i // BOARD_SIZE))
            self.b[i] *= NOISE_PROBS.get(cell, NOISE_PROBS[Cell.SPACE])[ni]
        self._norm()

    def update_distance(self, pos: Tuple[int,int], observed: int):
        for i in range(N):
            actual = abs(pos[0]-(i % BOARD_SIZE)) + abs(pos[1]-(i // BOARD_SIZE))
            p = sum(prob for off, prob in zip(DIST_ERR_OFF, DIST_ERR_P) 
                    if max(0, actual + off) == observed)
            self.b[i] *= p
        self._norm()

    def update_search(self, loc: Tuple[int,int], found: bool):
        if found: self.b[:] = 1.0 / N
        else:
            self.b[loc[1] * BOARD_SIZE + loc[0]] = 0.0
            self._norm()

    def _norm(self):
        s = self.b.sum()
        if s > 1e-15: self.b /= s
        else: self.b[:] = 1.0 / N

    def best_search_ev(self) -> Tuple[Tuple[int,int], float]:
        i = int(np.argmax(self.b))
        p = float(self.b[i])
        return (i % BOARD_SIZE, i // BOARD_SIZE), p * RAT_BONUS - (1.0 - p) * RAT_PENALTY

    def rat_ev(self) -> float:
        _, ev = self.best_search_ev()
        return ev

# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Logic (The Original "Heuristic 4")
# ═══════════════════════════════════════════════════════════════════════════════

# Weights per Phase
PHASE_WEIGHTS = {
    "opening": {"score": 0.2, "potential": 0.3, "mobility": 0.6, "danger": 0.3, "rat_ev": 0.1, "centrality": 0.4},
    "midgame": {"score": 1.0, "potential": 0.8, "mobility": 0.4, "danger": 0.2, "rat_ev": 0.3, "centrality": 0.2},
    "endgame": {"score": 1.5, "potential": 1.2, "mobility": 0.2, "danger": 0.5, "rat_ev": 0.8, "centrality": 0.1},
}

def evaluate(board: Board, belief: Optional[RatBelief]) -> float:
    pw, ow = board.player_worker, board.opponent_worker
    ppos, opos = pw.get_location(), ow.get_location()
    
    # Identify phase
    turn = board.turn_count
    phase = "opening" if turn < 10 else "midgame" if turn < 30 else "endgame"
    W = PHASE_WEIGHTS[phase]

    # Features
    score_delta = float(pw.get_points() - ow.get_points())
    
    def get_pot(pos):
        tot = 0.0
        for d in ALL_DIRS:
            cur = pos
            primed, space = 0, 0
            for _ in range(BOARD_SIZE-1):
                cur = loc_after_direction(cur, d)
                if not board.is_valid_cell(cur) or cur in (ppos, opos): break
                if board.get_cell(cur) == Cell.PRIMED: primed += 1
                else: break
            # Immediate and future discount
            if primed >= 2: tot += CARPET_POINTS_TABLE.get(primed, 0)
            tot += (primed * 0.2) 
        return tot

    pot_delta = get_pot(ppos) - get_pot(opos)
    
    def get_mob(pos):
        return sum(1 for d in ALL_DIRS if board.is_valid_cell(loc_after_direction(pos, d)) 
                   and not board.is_cell_blocked(loc_after_direction(pos, d)))
    
    mob_delta = get_mob(ppos) - get_mob(opos)
    
    dist = abs(ppos[0]-opos[0]) + abs(ppos[1]-opos[1])
    danger = -5.0 if dist == 0 else -3.0 if dist == 1 else 0.0
    
    rat_ev = belief.rat_ev() if belief else 0.0
    
    center = BOARD_SIZE // 2
    centrality = (abs(opos[0]-center) + abs(opos[1]-center)) - (abs(ppos[0]-center) + abs(ppos[1]-center))

    return (W["score"] * score_delta + W["potential"] * pot_delta + 
            W["mobility"] * mob_delta + W["danger"] * danger + 
            W["rat_ev"] * rat_ev + W["centrality"] * centrality)

# ═══════════════════════════════════════════════════════════════════════════════
#  Negamax Engine with TT & Move Ordering
# ═══════════════════════════════════════════════════════════════════════════════

class Negamax:
    def __init__(self, belief):
        self.belief = belief
        self.tt: Dict[int, dict] = {}
        self._dead = 0.0
        self.nodes = 0

    def best_move(self, board: Board, budget: float) -> Move:
        self._dead = time.perf_counter() + budget
        self.nodes = 0
        moves = self._get_ordered_moves(board)
        if not moves: return Move.search((BOARD_SIZE//2, BOARD_SIZE//2))
        
        best_overall = moves[0]

        for depth in range(1, 10): # Iterative Deepening
            if time.perf_counter() >= self._dead - 0.02: break
            try:
                _, mv = self._search(board, depth, -INF, INF)
                if mv: best_overall = mv
            except TimeoutError: break
            
        return best_overall

    def _get_ordered_moves(self, board: Board, tt_move: Move = None) -> List[Move]:
        raw = board.get_valid_moves(exclude_search=True)
        moves = [m for m in raw if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        
        def move_score(m):
            if tt_move and m == tt_move: return 1000
            if m.move_type == MoveType.CARPET: return 500 + m.roll_length
            if m.move_type == MoveType.PRIME: return 100
            return 0
            
        moves.sort(key=move_score, reverse=True)
        return moves

    def _search(self, board, depth, alpha, beta) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= self._dead: raise TimeoutError()
        
        alpha_orig = alpha
        h = ZOBRIST.get_hash(board)
        
        # 1. TT Lookup
        if h in self.tt and self.tt[h]['depth'] >= depth:
            entry = self.tt[h]
            if entry['flag'] == EXACT: return entry['val'], entry['move']
            elif entry['flag'] == LOWERBOUND: alpha = max(alpha, entry['val'])
            elif entry['flag'] == UPPERBOUND: beta = min(beta, entry['val'])
            if alpha >= beta: return entry['val'], entry['move']

        self.nodes += 1
        if depth == 0 or board.is_game_over():
            return evaluate(board, self.belief), None

        # 2. Search
        best_val = -INF
        best_move = None
        tt_move = self.tt[h]['move'] if h in self.tt else None
        moves = self._get_ordered_moves(board, tt_move)

        for m in moves:
            child = board.forecast_move(m)
            if not child: continue
            child.reverse_perspective()
            
            val = -self._search(child, depth - 1, -beta, -alpha)[0]
            
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, val)
            if alpha >= beta: break

        # 3. TT Store
        flag = EXACT
        if best_val <= alpha_orig: flag = UPPERBOUND
        elif best_val >= beta: flag = LOWERBOUND
        self.tt[h] = {'depth': depth, 'val': best_val, 'flag': flag, 'move': best_move}
        
        return best_val, best_move

# ═══════════════════════════════════════════════════════════════════════════════
#  The Agent Interface
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left=None, **kwargs):
        """
        Fixed signature to match engine expectations:
        1. board_state
        2. transition_matrix
        3. time_left (The func passed by the engine)
        **kwargs: catches any extra tournament-specific flags
        """
        self.belief = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.engine = Negamax(self.belief)
        self._prev_search = (None, False)
        # We store the time_left func if provided, though play() usually gets its own
        self.time_left_func = time_left 

    def play(self, board_state: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        noise, dist = sensor_data
        
        # 1. HMM Updates
        if self.belief:
            self.belief.predict()
            self.belief.update_noise(board_state, noise)
            self.belief.update_distance(board_state.player_worker.get_location(), dist)
            
            # Tracking searches
            # Opponent search result
            opp_loc, opp_hit = board_state.opponent_search
            if opp_loc:
                self.belief.update_search(opp_loc, opp_hit)
            
            # Our own last search result (don't double count)
            my_loc, my_hit = board_state.player_search
            if my_loc and my_loc != self._prev_search[0]:
                self.belief.update_search(my_loc, my_hit)
                self._prev_search = (my_loc, my_hit)

        # 2. Rat Search Decision: EV = 6p - 2. Patience decays over turns.
        if self.belief:
            sloc, sev = self.belief.best_search_ev()
            turns_played = MAX_TURNS_PER_PLAYER - board_state.player_worker.turns_left
            # Start strict (p >= 0.67), end lenient (p >= 0.33)
            patience = max(0.0, 2.0 - turns_played * (2.0 / 30.0))
            if sev >= patience:
                return Move.search(sloc)

        # 3. Negamax Execution
        turns_left = board_state.player_worker.turns_left
        # Budget: Reserve 0.2s buffer, split remaining time, cap at 5s.
        budget = max(0.05, min(5.0, (time_left() - 0.2) / max(1, turns_left)))
        
        return self.engine.best_move(board_state, budget)

    def commentate(self) -> str:
        if not self.belief:
            return "No belief tracking."
        top_loc, top_p = self.belief.top_n(1)[0] if hasattr(self.belief, 'top_n') else (None, 0)
        return f"Rat? {top_loc} ({top_p:.2f}) | Nodes: {self.engine.nodes} | TT: {len(self.engine.tt)}"