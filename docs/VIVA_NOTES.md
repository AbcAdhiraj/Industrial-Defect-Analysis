# Viva notes

Open each file in an editor before the viva and know where the named function is.
Numbers in the worked examples were checked by running the code.

---
## Module 1 - state-space search and CSP

**Five-line explanation**
1. A *problem* is five things: initial state, actions, result, goal test, step cost. The search code never looks inside a state.
2. Our problem: visit all required inspection checkpoints; state = (where I am, which checkpoints are done); cost = distance.
3. BFS/IDDFS find routes with the fewest steps, but every complete route has the same number of steps, so they ignore distance. UCS and A* use the cost `g`; A* adds a heuristic `h`.
4. Our `h` = minimum spanning tree of the remaining checkpoints + distance to the nearest one. It never overestimates, so A* is optimal but expands far fewer nodes.
5. Scheduling is a CSP: variables = checks, values = (station, slot); backtracking plus forward checking, MRV and AC-3 cut the search.

**Worked example (route)** DOCK (0,0); ALIGNMENT (3,4); SURFACE_SCAN (3,10); EDGE_PROFILE (9,4). ALIGNMENT must be first.
`h(start)` = MST over the 3 checkpoints (6 + 6 = 12) + nearest (5) = 17; the weak heuristic gives 5. A* returns
ALIGNMENT -> EDGE_PROFILE -> SURFACE_SCAN with cost 5 + 6 + 8.485 = 19.485, which equals brute force (17 <= 19.485: admissible).
**Worked example (CSP)** Map colouring of Australia with 3 colours has 18 solutions (6 for the mainland x 3 for Tasmania); with 2 colours `solve` returns `unsatisfiable`. 6-queens has 4 solutions, 8-queens 92.

**Complexity** BFS/UCS/A* time and space O(b^d); DFS O(b^l) time, O(b*l) space; IDDFS O(b^d) time, O(b*d) space. CSP backtracking O(d^n) worst case; AC-3 O(e*d^3); forward checking adds O(b*d) work per assignment but removes dead branches early.

**Likely questions**
1. *Why is A* optimal?* If h never overestimates, the first time the goal is popped every cheaper path has already been explored (f = g + h is a lower bound on any route through the node).
2. *Why is the MST heuristic admissible?* Any finishing route contains a path through all unvisited checkpoints, which contains a spanning tree (>= MST), plus a first leg >= nearest distance. Precedence only removes routes, so the true cost can only be larger. (Proof in `make_heuristics` docstring.)
3. *BFS vs UCS?* BFS minimises number of steps, UCS minimises cost; they differ when step costs differ - see the cost/optimal column in benchmark 3.
4. *What does forward checking do that plain backtracking does not?* After assigning X it deletes inconsistent values from neighbours' domains and backtracks at once if a domain empties.
5. *Why MRV?* "Fail first": choose the variable with the fewest legal values so dead ends are found near the root.

**Open:** `search.py` -> `best_first`, `a_star`; `routing.py` -> `make_heuristics`, `InspectionRoute.actions`; `csp.py` -> `solve`, `ac3`; `scheduling.py` -> `_pair_predicate`, `verify_schedule`.

---
## Module 2 - adversarial search (robustness check)

**Five-line explanation**
1. Question: if the sensors are slightly noisy, would the expert system still give the same label?
2. We model it as a game (a modelling choice): MAX = inspector, MIN = worst-case noise.
3. MAX may TRUST the label, send it to REVIEW (-0.5), or REMEASURE one borderline feature (cost -0.2, halves its noise band). MIN shifts a borderline reading by -delta, 0 or +delta.
4. TRUST pays +1 if the label survives the noise MIN chose, else -2. Minimax gives the value of optimal play; alpha-beta gives the same answer with fewer nodes.
5. The decision (TRUST / REMEASURE / REVIEW) is the first move of optimal play. True labels are used only afterwards, for evaluation.

**Worked example** Tree `[[3,5],[2,9]]`, MAX at the root, MIN below. MIN picks min(3,5)=3 and min(2,9)=2; MAX picks max = 3 (move 0). Minimax visits 7 nodes. Alpha-beta visits 6: after MIN sees the leaf 2 in the second branch, that branch is worth <= 2 < 3 (already guaranteed), so leaf 9 is pruned.

**Complexity** Minimax O(b^d) time, O(d) space. Alpha-beta O(b^(d/2)) with perfect ordering, O(b^d) worst case. Here b <= 12 and the depth limit is 4 plies plus at most 4 closing MIN plies.

**Likely questions**
1. *Why is alpha-beta correct?* It only skips branches that cannot change the value at the root (a bound already beats them), so value and first-best move equal minimax's (tested on 200 random trees and real plates).
2. *Where does pruning come from?* alpha = best MAX can force so far, beta = best MIN can force; if alpha >= beta the rest of the node is irrelevant.
3. *Is this a real two-player game?* No - it is a modelling choice: noise is treated as an adversary to get a worst-case (conservative) guarantee.
4. *What happens at the depth limit?* The node is scored as `cost + REVIEW`, the value MAX can always secure (conservative static evaluation).
5. *Limitation?* Only features of fired rules are considered; delta is a simple fraction of the training standard deviation.

**Open:** `robust_check.py` -> `minimax`, `alphabeta`, `RobustnessGame.moves/result/utility`, `borderline_features`.

---
## Module 3 - knowledge representation and expert system

**Five-line explanation**
1. Numeric measurements are turned into facts LOW / MEDIUM / HIGH using thresholds from training-set quartiles (one table, printed by `main.py`).
2. Knowledge is stored as frames (Plate, FaultType with `severity`, `typical_conditions`, `required_checks`), a semantic net (Z_Scratch is-a SurfaceDefect is-a Fault) and 30 IF-THEN rules.
3. Forward chaining fires every applicable rule: symptom rules first (derived facts), then fault rules; conflicts are resolved by priority, then specificity.
4. Backward chaining starts from a goal "fault == X" and builds a proof tree of rules and facts, or answers "not provable".
5. `explain` prints facts used, rules fired, conflicts resolved and the proof.

**Worked example** Facts: Y_Perimeter=LOW, X_Perimeter=HIGH, Steel_Plate_Thickness=LOW. Pass 1 fires D1 (ELONGATED_X) and D6 (THIN_PLATE). Then F7 (ELONGATED_X and THIN_PLATE, priority 3), F10 (priority 1) and the fallback F21 (priority 0) are applicable. F7 wins on priority -> Z_Scratch. Backward chaining on `fault = Z_Scratch` finds F7, then proves ELONGATED_X through D1 from the two perimeter facts.

**Complexity** Forward chaining O(P*R*C) (P passes, R rules, C conditions); backward chaining O(R*C) per goal on an acyclic base, space O(depth). Discretisation O(F) per plate.

**Likely questions**
1. *Forward vs backward chaining?* Forward is data-driven (facts -> conclusions); backward is goal-driven (conclusion -> needed facts). Use forward to classify, backward to verify or to ask which facts are missing.
2. *How are conflicts resolved?* Priority, then number of conditions, then rule order. See `forward_chain`.
3. *Frame vs semantic net?* A frame bundles slots of one object (with inheritance); a semantic net is a graph of labelled relations such as is-a.
4. *Where did the rules come from? Is it learned?* No learning: rules were written by hand from TRAIN distributions (`docs/RULES.md`). Thresholds are the only numbers taken from data.
5. *How good is it?* Run `benchmarks.py` and quote YOUR numbers, including the confusion matrix and the two baselines. Say honestly that `Other_Faults` is the hardest class.

**Open:** `kb.py` -> `Thresholds`, `build_fault_frames`, `SemanticNet.is_a`; `rules.py`; `inference.py` -> `forward_chain`, `backward_chain`, `explain`.
