# Rule-Based Defect Classification and Inspection Planning System

AI PBL project (BTech CSE) - steel-plate faults, **Python 3 standard library only**,
every algorithm written by hand. Scope: AI syllabus **Modules 1-3** (no machine
learning, neural nets, Bayesian nets or fuzzy logic). The system works on
*measured plate features*, not on images.

## The pipeline: one question per step

| Step | Question | Module | Technique |
|------|----------|--------|-----------|
| 1 CLASSIFY | Which fault does the plate have, and why? | 3 | frames, semantic net, rule base, forward/backward chaining, explanation |
| 2 CHECK | Is that answer stable if the sensor readings are noisy? | 2 | two-player game, minimax and alpha-beta |
| 3 PLAN | In what order are the required checkpoints visited? | 1 | state-space search (BFS/DFS/IDDFS/UCS/Greedy/A*) |
| 4 SCHEDULE | Which station and time slot runs each check? | 1 | constraint satisfaction (BT, FC, MRV, AC-3) |

The fault decides the required checks (each fault frame has a `required_checks`
slot), so step 1 feeds steps 3 and 4.

## Module -> file -> function

| Module | File | Main functions / classes |
|--------|------|--------------------------|
| data | `data_loader.py` | `load_dataset`, `validate_matrix`, `stratified_split`, `make_synthetic_matrix`, `try_download` |
| 3 | `kb.py` | `Thresholds` (discretisation), `Frame`, `build_fault_frames`, `SemanticNet`, `build_semantic_net`, `Rule` |
| 3 | `rules.py` | `RULES` (30 hand-written IF-THEN rules) |
| 3 | `inference.py` | `forward_chain`, `backward_chain`, `facts_needed`, `explain`, `classify` |
| 3 | `evaluation.py`, `rule_evidence.py` | accuracy, confusion matrix, 2 baselines; writes `docs/RULES.md` |
| 2 | `robust_check.py` | `RobustnessGame`, `minimax`, `alphabeta`, `borderline_features`, `robust_decision` |
| 1 | `search.py` | `Problem`, `breadth_first`, `depth_first`, `iterative_deepening`, `best_first`, `uniform_cost`, `greedy_best_first`, `a_star` |
| 1 | `routing.py` | `InspectionRoute`, `make_heuristics` (admissibility proof in the docstring), `brute_force_optimal` |
| 1 | `classic_problems.py` | `EightPuzzle`, `WaterJug`, `MissionariesCannibals` (engine sanity checks) |
| 1 | `csp.py` | `CSP`, `solve` (`bt`, `fc`, `fc_mrv`, `ac3_fc_mrv`), `ac3`, `n_queens`, `map_coloring` |
| 1 | `scheduling.py` | `make_instance`, `build_csp`, `verify_schedule`, `format_schedule` |
| all | `pipeline.py`, `main.py` | end-to-end demo |
| all | `benchmarks.py` | the four comparisons, CSVs in `results/` |
| docs | `docs/RULES.md`, `docs/VIVA_NOTES.md` | rule evidence; viva preparation |

## Build and run

No installation: only Python 3 (tested with 3.13).

```
python3 tests/test_all.py        # 14 test groups, exit code 1 on any failure
python3 main.py                  # demo (add --plates 10 --seed 1; --offline skips the download attempt)
python3 benchmarks.py            # full benchmarks, ~minutes; --quick for a smoke test
python3 rule_evidence.py --write # regenerate docs/RULES.md from the TRAIN split
```

## Data: real versus synthetic (read this first)

* Dataset: UCI *Steel Plates Faults* (1,941 plates, 27 features, 7 faults),
  https://archive.ics.uci.edu/dataset/198/steel+plates+faults
* **Place `Faults.NNA` in `data/`.** The loader also tries to download it and
  validates it (1941 rows, 34 columns, exactly one class flag per row).
* If the file is missing, a **SYNTHETIC generator with the same schema** is
  used so the code and tests can run. Everything printed or written then says
  **SYNTHETIC DATA** (CSV files have a `data_source` column). Synthetic numbers
  prove the code works; they say **nothing** about real steel plates. The
  generator's class profiles are invented by the project authors.
* When the build of this repo was made the UCI host was not reachable from the
  build machine (HTTP 403 from the sandbox proxy), so the committed `results/`
  and the numbers in `docs/RULES.md` come from **synthetic data**.
* **After adding the real file** re-run: `python3 rule_evidence.py --write`,
  *look at the per-rule precision*, revise weak rules in `rules.py` using the
  TRAIN split only, then run `tests/test_all.py` and `benchmarks.py` again.
* Protocol: stratified 70/30 split, seed 42. Discretisation thresholds
  (25%/75% train quantiles, table printed by `main.py`) and both baselines use
  the TRAIN split only. Rules were written from TRAIN distributions; the TEST
  split is used only for scoring and was not used to change any rule.

## Design decisions

* Rules are **two-layer**: layer 1 derives symptoms (`ELONGATED_X`, `THICK_PLATE`, ...)
  from discretised features, layer 2 concludes a fault. This gives a real chain
  for forward chaining and a proof tree of depth 2 for backward chaining.
* Conflict resolution: highest priority, then more conditions (specificity),
  then earlier rule. `Other_Faults` is the closed-world fallback (priority 0).
* **Robustness check is a modelling choice** (see header of `robust_check.py`):
  MAX = inspector (TRUST / REVIEW / REMEASURE f), MIN = worst-case noise
  (+/- delta on a borderline feature). delta = 10% of the feature's TRAIN
  standard deviation. At most 4 borderline features. Payoffs and depth limit as
  specified (REVIEW -0.5, TRUST +1 / -2, REMEASURE -0.2, depth 4).
  True labels are never used inside the game, only to score it afterwards.
* Fault taxonomy, severities, `required_checks`, sensors, check precedence and
  station equipment are **engineering assumptions** made for the project.
* If the check says REVIEW, a `MANUAL_REVIEW` checkpoint is added to the route.
* Search uses graph search (visited set) where it is correct; depth-limited DFS
  uses only a path cycle check (a global visited set is unsound with a depth limit).
* Brute force exists only to verify A*/UCS (K <= 8) in the tests.

## Limitations

* Hand-written rules reach modest accuracy; weak results are reported, not hidden.
  `Other_Faults` (a catch-all class) is intrinsically hard for rules.
* The robustness check measures whether the label is *stable* under noise, not whether it is *correct*. In the synthetic run it did **not** raise accuracy on TRUST plates above the all-plates accuracy (see `results/benchmark_output.txt`); check what happens on the real data and report it honestly.
* The game only considers features of rules that *fired*; a noisy reading that
  would switch on a different rule is not modelled. The noise model is simple.
* The route and schedule instances (coordinates, stations, sensors) are
  simulated; nothing here is connected to real equipment.
* In benchmarks "mean of 5 runs" means: classifier/robustness results are
  deterministic (only timings are averaged); routes and scheduling use 5
  different seeded instance sets, identical for all methods within a run.
* Scheduling instances can be unsatisfiable or too hard for a solver within the
  node cap; such cases are reported (`unsatisfiable` / `capped`), never hidden.
* The download code path (`try_download`, zip handling) could not be tested
  from the build machine because the network was blocked.

## Measured results (fill in on the REAL data)

Copy the tables printed by `python3 benchmarks.py` here after running with the
real `Faults.NNA`. (The CSVs in `results/` currently hold SYNTHETIC output.)

| Item | Your measured value |
|------|---------------------|
| Expert-system test accuracy | _fill in_ |
| Majority baseline / best single-feature baseline | _fill in_ |
| Accuracy on TRUST plates / on all plates / REVIEW rate | _fill in_ |
| Alpha-beta node saving versus minimax | _fill in_ |
| A* vs BFS nodes at K = 9 | _fill in_ |
| CSP: backtracking vs FC+MRV nodes (largest instance finished) | _fill in_ |
