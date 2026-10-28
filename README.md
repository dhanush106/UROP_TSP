# TSP_XGBOOST

## 1. Project purpose

`TSP_XGBOOST` is a native C++ implementation of machine-learning-based
edge reduction for the symmetric Travelling Salesperson Problem (TSP). It
uses an XGBoost binary classifier to estimate which undirected graph edges
belong to an optimal or near-optimal tour. Predicted edges are retained, and
the edges of a sampled feasible tour are restored before the reduced TSP is
solved with the open-source HiGHS mixed-integer optimization solver.

The application has two commands:

1. `train`: reads labeled TSP instances, creates graph-level training,
   validation, and test splits, trains the edge classifier, and saves the
   model, its metadata, the datasets, and metrics.
2. `reduce`: reads one new TSP instance and a trained model, calculates edge
   features, prunes edges, restores a sampled feasible tour, solves the
   reduced TSP with HiGHS, and writes the tour and benchmark results.

This is an experimental reproduction/extension of the *problem-reduction
workflow*, not a bit-for-bit reproduction of the paper's MLPR experiments.
The research paper uses cost-sensitive SVM models and reports experiments
with multiple instance groups, random seeds, and solvers. This project uses
XGBoost and HiGHS. Section 12 explains the similarities, differences, and
what can and cannot be concluded from the current results.

## 2. Important result in brief

The currently saved Berlin52 result in `results/` records:

| Quantity | Current recorded result |
|---|---:|
| Original undirected edges | 1,326 |
| Retained undirected edges | 268 |
| Edges removed | 1,058 (79.79%) |
| Best sampled-tour cost before solving | 23,704 |
| HiGHS reduced-problem objective | 7,542 |
| HiGHS best bound | 7,542 |
| HiGHS relative gap | 0 |
| HiGHS status | Optimal |
| Recorded total run time | 4.935 seconds |

The objective 7,542 matches the known optimal value for TSPLIB Berlin52.
The saved HiGHS tour is a real incumbent: the solver proved it optimal for
the reduced MILP, the code checks that it is a Hamiltonian cycle, and it
checks the reported objective against the cost recomputed from that tour.
This is strong evidence for this particular Berlin52 run. It is not evidence
of a general success rate or of a speedup over the full graph: this project
has not established those claims with a repeated benchmark study.

Training metrics saved in `data/splits/edge_metrics.csv` are:

| Split | Instances | Edge rows | Average precision | Edge recall | Edge precision | Retained fraction | Complete-tour fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 133 | 658,350 | 0.73152 | 0.99737 | 0.11072 | 0.18198 | 0.74436 |
| Validation | 28 | 138,600 | 0.72553 | 0.99500 | 0.11185 | 0.17971 | 0.57143 |
| Test | 29 | 143,550 | 0.71857 | 0.99586 | 0.10961 | 0.18354 | 0.68966 |

These are edge-classification and tour-retention metrics on the included
MATILDA `LKCChard` data. They are **not** the same as the paper's optimality
gap and solver speedup metrics. Interpret them accordingly.

## 3. Repository layout

```text
TSP_XGBOOST/
  CMakeLists.txt
  README.md
  .gitignore
  data/
    train/                    Included labeled training instances
      *.tsp                   TSPLIB graph instance
      *.sol                   Corresponding known tour
    splits/                   Generated training artifacts (created by train)
      train_data.csv
      validation_data.csv
      test_data.csv
      instance_splits.csv
      dataset_summary.csv
      edge_metrics.csv
      training_history.csv
  artifacts/
    tsp_xgboost.json           Saved XGBoost model
    tsp_xgboost.json.meta      Model settings and reduction threshold
  src/
    main.cpp                   Command-line parsing and program entry point
    graph.h / graph.cpp         TSPLIB graph and tour parsing
    features.h / features.cpp   Sampling, edge labels/features, tour cost
    xgboost_api.h / .cpp        Dynamic loading and calls to XGBoost C API
    ml_pipeline.h / .cpp        Training, metrics, inference, result writing
    solver.h                    Solver result contract
    solver_highs.cpp            HiGHS MILP model and solution validation
    solver_stub.cpp             Explicit no-solver implementation
  tests/
    test_core.cpp               Parser and deterministic-feature tests
    test_solver.cpp              Four-city exact-solver integration test
  results/
    berlin52_res_obj.csv         Berlin52 objective/bound/gap/status
    berlin52_res_time.csv        Berlin52 phase timings
    berlin52_res_sol.csv         Berlin52 solver tour
    berlin52/                    Berlin52 edge and summary files
```

The included training pairs were copied from
`..\tsp\TSP_MATILDA\LKCChard\`. There are 190 `.tsp` files and 190 matching
tour files. The original `tsp` directory is not changed by this project.
Generated build directories, model artifacts, split data, and results may
be ignored by Git according to `.gitignore`; ignoring them does not delete
them from your computer.

### Source-file responsibilities

- **`src/main.cpp`** parses `train` and `reduce` options, validates required
  arguments, calls the pipeline, and reports errors with a nonzero exit code.
- **`src/graph.cpp`** reads a TSPLIB graph into a symmetric distance matrix
  and reads a tour as a permutation of city IDs. The internal city IDs are
  zero-based; CSV and tour-file output is one-based.
- **`src/features.cpp`** samples random feasible tours, computes the best
  sampled tour and its cost, calculates six edge features, and provides the
  tour-cost calculation used to validate solver results.
- **`src/xgboost_api.cpp`** loads the XGBoost shared library at runtime and
  calls the XGBoost C API. The project does not require Python or XGBoost
  headers to run, but it does require a compatible native XGBoost DLL/shared
  library.
- **`src/ml_pipeline.cpp`** makes instance-level splits, builds labels,
  writes datasets, trains and evaluates XGBoost, chooses the threshold,
  performs inference and graph reduction, and writes result files.
- **`src/solver_highs.cpp`** creates a directed-arc TSP MILP for the retained
  graph, gives HiGHS a sampled-tour initial solution, solves the model, and
  validates any returned Hamiltonian cycle and objective.
- **`src/solver_stub.cpp`** returns `not_run` when configured with
  `-DTSP_XGBOOST_WITH_HIGHS=OFF`. It deliberately does not fabricate an
  objective or tour.
- **`tests/test_core.cpp`** checks a small TSPLIB parser/feature case and
  deterministic sampling for a fixed seed.
- **`tests/test_solver.cpp`** solves a four-city square TSP and checks the
  optimal objective, bound, and gap.

## 4. End-to-end workflow

### 4.1 Training data and labels

Each `.tsp` instance is paired with an optimal-tour file (`.sol`,
`.opt.tour`, or `.tour`). For an instance with `n` cities:

- The graph has `n(n-1)/2` undirected edges.
- The tour contributes `n` positive edge labels.
- Every other graph edge is labeled negative.
- Each edge row contains the instance name, endpoint IDs, raw edge cost,
  binary label, and the six edge features.

The label means “appears in the supplied known tour,” not “is proven to be
the only edge that could occur in any optimal tour.” If an instance has
multiple distinct optimal tours but only one is supplied, edges from another
equally optimal tour will be labeled negative.

### 4.2 Instance-level splitting

The program sorts the discovered `.tsp` paths, shuffles instance indices
using the training seed, and assigns whole instances to approximately:

- 70% training
- 15% validation
- 15% test

Edges from the same graph therefore do not occur in more than one split.
With the included 190 instances, the saved split is 133 training, 28
validation, and 29 test instances. Because each included instance has 100
cities and 4,950 undirected edges, the resulting total is 940,500 edge rows:
19,000 positive and 921,500 negative rows.

The training/validation/test assignment is reproducible with the same input
file set and seed. The test split is held out from model and threshold
selection; it is evaluated only after those choices have been made.

### 4.3 Model training and threshold selection

The XGBoost model is a gradient-boosted decision-tree binary classifier:

| Setting | Value |
|---|---|
| Objective | `binary:logistic` |
| Tree method | `hist` |
| Evaluation metric | `aucpr` (area under precision-recall curve) |
| Maximum depth | 5 |
| Learning rate (`eta`) | 0.05 |
| Row subsampling | 0.8 |
| Column subsampling per tree | 0.9 |
| Minimum child weight | 5 |
| L2 regularization (`lambda`) | 1.0 |
| Positive-class weight | Training negatives / training positives |
| Maximum rounds by default | 1,200 |
| Early-stopping patience by default | 50 rounds |

The pipeline updates the model one round at a time and measures validation
average precision after every round. It saves the checkpoint with the best
validation average precision and stops after the configured patience without
improvement. The saved model is reloaded before the split metrics are
calculated.

The classification threshold is selected from validation positive-edge
scores to meet the requested validation edge-recall target (default 0.995).
That threshold is then applied unchanged to the training, validation, and
test score vectors. XGBoost outputs probabilities; an edge is initially
predicted for retention when its score is greater than or equal to this
threshold.

The model file and its `.meta` file form one logical artifact. The metadata
contains the threshold, best iteration, validation average precision, recall
target, seed, sampling factor, model settings, positive-class weight, and
feature order. Do not copy the JSON model without its metadata file.

### 4.4 Inference, graph reduction, and solving

For a new instance, the reducer:

1. Parses the graph and computes the same six features for each undirected
   edge.
2. Loads the model and metadata, scores each edge, and retains edges meeting
   the stored threshold.
3. Restores every edge in the best of the sampled tours. These restored
   edges ensure the reduced graph contains at least one feasible Hamiltonian
   tour, even if the classifier would otherwise remove one of its edges.
4. Passes the retained graph to HiGHS as a MILP and supplies that sampled
   tour as an initial feasible solution.
5. Validates any incumbent as one Hamiltonian cycle and recomputes its cost
   from the input graph before writing the result.

The sampled-tour repair is a feasibility safeguard, not a proof that the
original globally optimal tour remains in the graph. The HiGHS proof of
optimality, when reported, applies to the **reduced** problem. A reduced
problem optimum is not automatically an original-problem optimum. Comparing
the objective to an independently known optimum, as for Berlin52, provides
additional evidence for that instance.

## 5. Edge features

The feature order is shared by training, model metadata, and inference:

1. **`correlation_normalized`** — the correlation between sampled-tour
   membership of an edge and the sampled tour costs. Edges associated with
   lower tour costs tend to have negative correlation. The value is divided
   by the most negative edge correlation when that value is nonzero.
2. **`ranking_normalized`** — for each sample tour, the edge receives
   reciprocal rank `1/r`, where `r` is that tour's cost rank (best tour has
   rank 1). The edge scores are summed and divided by the largest graph edge
   score.
3. **`cost_vs_min_i`** — `(edge cost - minimum incident cost at i) /
   incident-cost range at i`.
4. **`cost_vs_min_j`** — the corresponding endpoint-relative quantity at `j`.
5. **`cost_vs_mean_i`** — `(edge cost - mean incident cost at i) /
   incident-cost range at i`.
6. **`cost_vs_mean_j`** — the corresponding endpoint-relative quantity at `j`.

The feature set matches the paper's structure: two global sample-derived
measures and four endpoint-relative edge-cost features. The implementation
guards degenerate normalization ranges to keep feature values finite. For a
zero cost range, the denominator is replaced by at least 1.

### Sampling budget and complexity

`--sample-factor q` means the number of sampled tours is `m = q × n`, not
simply `q` tours. The default `q=100` therefore matches the paper's
recommended sampling budget `m=100n`. A fixed seed makes a run's sampling
reproducible for the same executable and library environment.

The implementation stores sampled routes to rank them, so its work includes
sampling/edge accumulation, sorting `m` tour costs, and computing graph-wide
edge data. Approximate memory use is proportional to `m×n + n²`; at the
default `m=100n`, that can become substantial for large graphs. Reduce the
sample factor for a smoke test or large input, understanding that changing it
also changes the statistical features and results.

## 6. Training artifacts and metrics

By default, the training command writes these files under `data/splits/`:

| File | Contents |
|---|---|
| `train_data.csv` | One labeled edge row per training graph edge |
| `validation_data.csv` | One labeled edge row per validation graph edge |
| `test_data.csv` | One labeled edge row per test graph edge |
| `instance_splits.csv` | Instance name and its assigned split |
| `dataset_summary.csv` | Instance, row, positive-edge, and negative-edge counts |
| `edge_metrics.csv` | Split-level average precision, classification and retention metrics |
| `training_history.csv` | Per-round validation AP, best AP, best iteration, and improvement flag |

The split CSV columns are:

```text
instance,city_i,city_j,edge_cost,label,
correlation_normalized,ranking_normalized,cost_vs_min_i,cost_vs_min_j,
cost_vs_mean_i,cost_vs_mean_j
```

### Metric definitions

- **Average precision (AP):** ranking quality across scored edge rows. Validation
  AP chooses the model round. AP is not edge accuracy or tour optimality.
- **Edge recall:** `TP / (TP + FN)`, the fraction of supplied optimal-tour
  edges predicted for retention before repair.
- **Edge precision:** `TP / (TP + FP)`, the fraction of predicted-retained
  edges that are positive under the supplied tour labels.
- **Retained-edge fraction:** predicted-positive edge rows divided by all
  edge rows before sampled-tour repair.
- **Complete-tour fraction:** fraction of instances in the split for which
  every edge of the supplied tour is predicted positive **before** repair.
- **TP / FP / FN / TN:** edge-level confusion-matrix counts.

The labels are highly imbalanced: on a 100-city complete graph, just 100 of
4,950 edges are in the labeled tour. Therefore, a threshold chosen for 99.5%
recall can yield low precision: preserving almost all positive edges requires
keeping many negative edges too. Recall is measured edge-by-edge; it does not
mean 99.5% of whole optimal tours are preserved. A tour requires all `n` of
its edges, so a small number of false negatives can make complete-tour
retention notably lower than edge recall.

### Current saved split metrics

The values below come from the current `data/splits/edge_metrics.csv`, using
the model whose metadata is in `artifacts/tsp_xgboost.json.meta`:

| Split | AP | Recall | Precision | Retained fraction | Complete-tour fraction | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.731516 | 0.997368 | 0.110720 | 0.181981 | 0.744361 | 13,265 | 106,542 | 35 | 538,508 |
| Validation | 0.725527 | 0.995000 | 0.111852 | 0.179711 | 0.571429 | 2,786 | 22,122 | 14 | 113,678 |
| Test | 0.718566 | 0.995862 | 0.109614 | 0.183539 | 0.689655 | 2,888 | 23,459 | 12 | 117,191 |

The similar train, validation, and test AP values are a useful check against
obvious graph-level overfitting in this particular split. They do not prove
generalization to other instance categories, graph sizes, asymmetric TSP, or
other problem variants.

The currently saved model metadata records:

```text
best_iteration=93
validation_average_precision=0.7255273349
threshold=0.09189850837
recall_target=0.995
sample_factor=100
scale_pos_weight=48.5
```

Re-training overwrites the model, metadata, split CSVs, and metrics at the
selected paths. Save a copy elsewhere if you need to preserve a prior
experiment.

## 7. Reduction output files

For an input named `berlin52.tsp` and output directory `results`, the reducer
writes:

| Output | Meaning |
|---|---|
| `results/berlin52_res_obj.csv` | Directed-arc counts, sampled-tour cost, solver objective, bound, gap, status, optimal flag |
| `results/berlin52_res_time.csv` | Read, sampling, reduction, solver, and total seconds |
| `results/berlin52_res_sol.csv` | `TOUR_SECTION`, one-based city IDs, and `EOF` if an incumbent exists |
| `results/berlin52/edge_predictions.csv` | Every original edge, score, threshold decision, repair decision, and six features |
| `results/berlin52/reduced_edges.csv` | Retained undirected edges, one-based endpoint IDs, cost, score |
| `results/berlin52/best_sampled_tour.txt` | Best sampled tour, one-based city IDs |
| `results/berlin52/summary.txt` | Threshold, reduction counts/fractions, phase times, status, objective, bound, gap |

The three root-level benchmark files are overwritten if the same instance is
run again to the same output directory. With the no-solver build, the
solution file contains only `TOUR_SECTION`; solver objective/bound/gap are
left unavailable rather than replaced with fake values.

### Current Berlin52 files

The following table transcribes the **exact values saved for the Berlin52
run** in `results/berlin52_res_obj.csv`, `results/berlin52_res_time.csv`,
and `results/berlin52/summary.txt`. Values are shown at the precision stored
in those files, not rounded for presentation.

| Category | Result field | Exact saved value |
|---|---|---:|
| Graph | Original undirected edges | 1326 |
| Graph | Original directed arcs | 2652 |
| Reduction | Retained undirected edges | 268 |
| Reduction | Retained directed arcs | 536 |
| Reduction | Retained edge fraction | 0.2021116139 |
| Reduction | Removed edge fraction | 0.7978883861 |
| Reduction | Repaired sampled-tour edges | 37 |
| Sampling | Best sampled-tour objective | 23704 |
| HiGHS | Status | `Optimal` |
| HiGHS | Has feasible solution | `true` |
| HiGHS | Proven optimal for reduced MILP | `true` |
| HiGHS | Solver objective | 7542 |
| HiGHS | Best bound | 7542 |
| HiGHS | Relative gap | 0 |
| Timing | Graph reading (seconds) | 0.0013225 |
| Timing | Sampling (seconds) | 0.008125 |
| Timing | Reduction (seconds) | 0.322795 |
| Timing | HiGHS solve (seconds) | 4.6021526 |
| Timing | Total (seconds) | 4.9347809 |

This is one saved run, not an average over repeated seeds. `Optimal` proves
the result for the reduced MILP; the objective `7542` also matches the known
TSPLIB Berlin52 optimum. The total time is not a measured speedup: there is
no corresponding full-graph HiGHS baseline in these result files.

The saved `results/berlin52_res_obj.csv` contains:

```text
original_directed_arcs,sampled_tour_objective,reduced_directed_arcs,solver_objective,best_bound,relative_gap,solver_status,is_optimal
2652,23704,536,7542,7542,0,Optimal,1
```

The corresponding `summary.txt` records:

```text
retained_undirected_edges=268
retained_edge_fraction=0.2021116139
removed_edge_fraction=0.7978883861
repaired_tour_edges=37
solver_status=Optimal
solver_objective=7542
solver_best_bound=7542
solver_relative_gap=0
```

The current timing file reports approximately 4.935 seconds total, including
about 4.602 seconds in HiGHS. These are measurements from one run on the
current local build and machine; runtime changes with compiler, hardware,
threads, solver version, and system load. Do not treat this one timing as a
benchmark comparison.

## 8. HiGHS TSP model

For every retained undirected edge `{i,j}`, `solver_highs.cpp` adds both
directed arcs `(i,j)` and `(j,i)`. It minimizes the sum of selected arc costs.
It imposes:

- exactly one outgoing arc from each city;
- exactly one incoming arc to each city;
- integer order variables for cities other than the fixed depot;
- Miller-Tucker-Zemlin (MTZ) constraints to rule out disconnected subtours.

The best sampled tour is rotated to begin at city 0 before its directed arcs
and order values are supplied to HiGHS. The code checks that the returned
incumbent has one outgoing edge per city, returns to its start after visiting
every city exactly once, and has a cost matching the solver-reported
objective. A `Optimal` solver status means HiGHS proved optimality for that
MILP. If the time limit is reached, a feasible incumbent may exist without
an optimality proof; inspect status, bound, and gap.

The MTZ model is a pedagogical and generally applicable exact formulation,
but it is not a specialized high-performance TSP solver such as Concorde.
Large reduced instances may still be difficult for it.

## 9. Research-paper analysis and result comparison

### 9.1 What the paper does

The supplied paper is:

> Yuan Sun, Andreas Ernst, Xiaodong Li, and Jake Weiner, “Generalization of
> Machine Learning for Problem Reduction: A Case Study on Travelling Salesman
> Problems,” *OR Spectrum* 43 (2021), 607–633. The preprint is arXiv:2005.05847.

The paper adapts its Machine Learning for Problem Reduction (MLPR) method to
TSP. For solved training graphs, it labels edges in the optimal tour positive
and the other edges negative. It extracts the same broad six-feature design
used here:

- four local, endpoint-relative edge-cost features;
- a ranking-based edge score from stochastic feasible tours;
- a Pearson-correlation score between edge membership and tour objective.

At inference, the classifier predicts edges to retain and the optimization
problem is solved on the resulting sparse graph. The paper explicitly keeps
the edges of the best sampled tour so the reduced problem remains feasible.

The paper uses cost-sensitive SVM classification, not XGBoost. It discusses
RBF-kernel L1-SVM for smaller datasets and linear L2-SVM for very large
training sets. Its positive-class penalty is increased because a false
negative on a true tour edge is much more harmful than keeping an unnecessary
edge. Its sampling experiments use `m=100n` random feasible tours.

### 9.2 Experiments and headline findings in the paper

The paper evaluates generalization in three main ways:

1. **Different instance characteristics.** It uses seven MATILDA categories,
   each with 190 symmetric 100-city instances. In the main within-category
   setup, it trains on 50 instances and tests on 140 unseen instances; it
   also transfers models across the different categories. It repeats
   sampling/reduction 25 times and compares optimality gaps and remaining
   problem sizes after exact solving with Concorde. Its Table 1/2 results
   show that, depending on the training category, the model can trade
   reduced size against optimality gap. For example, MLPR trained on
   `LKCChard` (I6) and evaluated on I6 reports a 0.18% average gap and
   22.80% remaining problem size. Training on the random I7 category gives
   about 85% pruning and 0.44% average gap across the seven categories.
2. **Different problem sizes.** It trains on 190 randomly generated
   100-city instances, then tests on random instances from 200 through 2,000
   cities. The paper reports zero optimality gap in those experiments and
   notes the test instances share characteristics with training data.
3. **Different problem types and generic-solver preprocessing.** It tests
   symmetric TSP, asymmetric TSP, and sequential ordering problems, and
   examines CPLEX as a generic solver. For its six-instance CPLEX comparison,
   it reports speedups for Att48, Berlin52, Eil51, Gr48, Hk48, and Swiss42
   of 1.93x, 3.92x, 1.96x, 2.74x, 1.78x, and 3.01x respectively. Those are
   CPLEX time ratios from the paper's experiment setup, not absolute times
   that can be compared directly to this project's HiGHS runtime.

The paper's conclusions are not that pruning always preserves an optimum.
It describes a heuristic whose generalization gets worse when test instances
differ more from training data. It reports both residual problem size and
optimality gap, and its evaluations use repeated trials for stochastic
sampling.

### 9.3 Similarities

This implementation follows the paper's central approach:

- It classifies graph edges using optimal-tour training labels.
- It uses the two sample-derived statistics and four local cost features.
- The default sampling budget is `m=100n`.
- It removes predicted-negative edges and restores the best sampled tour.
- It uses the learned classifier as a preprocessing/reduction step before an
  optimization solver.
- Its current Berlin52 run removes 79.79% of edges, within the broad 70–80%
  reduction range discussed in the paper's comparison of methods.
- Its Berlin52 reduced-problem solution has the known TSPLIB optimum cost.

### 9.4 Differences that prevent a direct reproduction claim

The current results are **not directly numerically comparable** to the
paper's results:

| Aspect | Paper | This project |
|---|---|---|
| Classifier | Cost-sensitive SVM (RBF or linear, by experiment) | XGBoost histogram gradient-boosted trees |
| Positive penalty | SVM penalty parameter `m`, including reported settings such as 10 or 100 | `scale_pos_weight` from training class ratio plus validation-selected probability threshold |
| Training setup | Often 50 training instances per category; experiments also transfer across categories | 190 `LKCChard` instances split into 133/28/29 train/validation/test |
| Generalization test | Multiple MATILDA categories, sizes, and problem types | Held-out instances from one MATILDA category plus a Berlin52 run |
| Stochastic repetition | 25 sampling/reduction runs in several evaluations | The shown Berlin52 output is one seed/run |
| Main evaluation | Optimality gap, reduced size, and in one experiment CPLEX speedup | Edge AP/recall/precision, complete-tour retention, and one HiGHS reduced solve |
| TSP solver | Concorde for symmetric TSP experiments; CPLEX for other/generic-solver experiments | HiGHS with a directed MTZ MILP |

This project's held-out test split is meaningful for unseen graphs from the
same `LKCChard` category, but it does not reproduce the paper's full
cross-category evaluation. The paper's Table 3 Berlin52 number is a 3.92x
CPLEX speedup from solving original and reduced instances under its specified
conditions. This project's 4.935-second total is one XGBoost/HiGHS run and
has no matching full-graph HiGHS baseline, so it cannot establish a speedup
or verify that 3.92x result.

### 9.5 Are this project's results “matching” the paper?

**They are qualitatively consistent in one useful respect, but they do not
constitute a reproduction of the paper's reported results.**

- **Consistent signal:** Berlin52 was reduced to about 20.21% of its edges,
  and the resulting reduced model yielded the known optimal cost 7,542.
  That demonstrates that this particular trained XGBoost model, threshold,
  sample, and HiGHS solve worked well on this instance.
- **Comparable reduction magnitude only:** the approximately 79.79% pruning
  is in the same broad range as some of the paper's results. The percentages
  alone do not show equivalent generalization or solver improvement.
- **Not comparable classifier results:** test AP and edge recall are not the
  paper's optimality-gap metric. The paper evaluates whether the reduced
  optimization problem preserves optimal solution quality after solving.
- **Not comparable speed:** no original full-graph solve with the same HiGHS
  formulation, machine, threads, and time limit is recorded. Therefore there
  is no speedup ratio to compare with the paper.
- **Not yet evidence of broad generalization:** the model was trained and
  split on one MATILDA category, and the displayed Berlin52 result is a
  single trial. Other categories, sizes, seeds, and problem variants have
  not been benchmarked here.

For a fair paper-style evaluation, run multiple fixed seeds per test instance,
record reduced size and solver objective, compare each result with the known
original optimum to compute percentage optimality gap, and benchmark a
full-graph HiGHS solve against reduced-graph HiGHS using the same machine,
formulation, threads, time cutoff, and accounting rules. Also use separate
training categories and held-out categories/sizes if the objective is to
test the paper's generalization claims.

### Paper reference

Sun, Y., Ernst, A., Li, X., and Weiner, J. “Generalization of Machine Learning
for Problem Reduction: A Case Study on Travelling Salesman Problems.”
*OR Spectrum*, 43, 607–633 (2021).
[Article DOI](https://doi.org/10.1007/s00291-020-00604-x) ·
[arXiv abstract and preprint](https://arxiv.org/abs/2005.05847)

## 10. Requirements and build

### Requirements

- Windows 10/11 (the commands below use PowerShell).
- CMake 3.18 or newer.
- A C and C++14 compiler toolchain.
- Git and network access on the first CMake configure, unless the HiGHS
  FetchContent source is already cached.
- A native XGBoost shared library whose architecture matches the executable.

HiGHS v1.10.0 is fetched from its upstream Git repository and built by CMake;
HiGHS examples and tests are disabled in the dependency build. No CPLEX
installation, license, headers, or libraries are required. XGBoost is loaded
dynamically through the native C API, so Python and XGBoost headers are not
needed at application runtime.

### Build with the verified x64 MinGW toolchain

The following is the explicit toolchain used to validate this workspace. The
Code::Blocks MinGW compiler directory is put first on `PATH` so CMake does
not accidentally use a different MinGW `make.exe`.

```powershell
cd C:\Users\sarpo\OneDrive\Desktop\UROP_TSP\TSP_XGBOOST

$mingw = 'C:\Program Files\CodeBlocks\MinGW\bin'
$env:PATH = "$mingw;$env:PATH"

cmake -S . -B build-highs -G 'MinGW Makefiles' `
  -DCMAKE_MAKE_PROGRAM="$mingw\mingw32-make.exe" `
  -DCMAKE_C_COMPILER="$mingw\gcc.exe" `
  -DCMAKE_CXX_COMPILER="$mingw\g++.exe" `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-highs --parallel 8
ctest --test-dir build-highs --output-on-failure
```

The executable is `build-highs\bin\TSP_XGBOOST.exe`. The CTest suite should
report two passing tests, including the HiGHS square-TSP solve with objective
4.

If you use Visual Studio or another compiler, use the toolchain supported by
your local CMake installation. Do not reuse the MinGW build directory with a
different compiler: make a new build directory for each generator/toolchain.

### Optional build without HiGHS

This mode supports ML training and reduced-graph export only. It reports
`solver_status=not_run` and does not produce a solver tour or objective.

```powershell
cmake -S . -B build-no-solver -G 'MinGW Makefiles' `
  -DCMAKE_MAKE_PROGRAM="$mingw\mingw32-make.exe" `
  -DCMAKE_CXX_COMPILER="$mingw\g++.exe" `
  -DTSP_XGBOOST_WITH_HIGHS=OFF `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-no-solver --parallel 8
ctest --test-dir build-no-solver --output-on-failure
```

## 11. Locate XGBoost and run commands

### 11.1 Set the DLL path in PowerShell

The literal example `C:\path\to\xgboost.dll` is only a placeholder. Do not
assign that placeholder as the library path. Search a standard per-user
Python installation for the actual `xgboost.dll`:

```powershell
$xgbDll = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python" `
  -Filter xgboost.dll -Recurse -ErrorAction SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName

if (-not $xgbDll) {
  throw 'xgboost.dll was not found. Install XGBoost or locate its native DLL.'
}

$env:XGBOOST_LIB = $xgbDll
Test-Path $env:XGBOOST_LIB
```

`Test-Path` must return `True`. If XGBoost is installed outside that folder,
set `$env:XGBOOST_LIB` to its actual DLL path. The path must point to a native
shared library (on Windows, normally `xgboost.dll`), not to a Python file or
the XGBoost package directory. The DLL architecture must match the executable
(for the verified build, x64).

The environment variable lasts for the current PowerShell session. Set it
again in a new terminal session. You can avoid the environment variable by
passing `--xgboost-lib "C:\actual\path\xgboost.dll"` to either command.

### 11.2 Train the model

From the project directory, after building and setting the DLL path:

```powershell
.\build-highs\bin\TSP_XGBOOST.exe train `
  --data data\train `
  --dataset-output data\splits `
  --model artifacts\tsp_xgboost.json `
  --sample-factor 100 `
  --seed 12345 `
  --rounds 1200 `
  --early-stopping 50 `
  --recall-target 0.995 `
  --threads 4
```

This reads the 190 included instance/tour pairs and may take some time because
it calculates sample-derived features for all training graphs. A successful
run writes the split CSV files, metrics and history, the model, and
`artifacts\tsp_xgboost.json.meta`.

To run a **functional smoke test**, create a separate folder containing at
least four matching `.tsp`/tour pairs, and try `--sample-factor 1 --rounds 2`.
Do not treat metrics from four instances or two boosting rounds as model
quality results. Use a separate model path and dataset-output directory if
you want to preserve the saved full-training artifacts.

### 11.3 Reduce and solve Berlin52

After training, or using the currently saved model and matching `.meta` file:

```powershell
.\build-highs\bin\TSP_XGBOOST.exe reduce `
  --tsp ..\tsp\TSPLIB\berlin52.tsp `
  --model artifacts\tsp_xgboost.json `
  --output results `
  --sample-factor 100 `
  --seed 54321 `
  --solver-time-limit 1000 `
  --solver-threads 8
```

The solver time-limit value is in seconds. A positive thread count is passed
to HiGHS. `--solver-threads 0` leaves the choice to the solver. The XGBoost
library path is obtained from `XGBOOST_LIB` unless `--xgboost-lib` overrides
it.

Check the run and result files:

```powershell
Get-Content results\berlin52_res_obj.csv
Get-Content results\berlin52_res_time.csv
Get-Content results\berlin52\summary.txt
Get-Content results\berlin52_res_sol.csv
Import-Csv results\berlin52\reduced_edges.csv | Measure-Object
```

Display CLI help:

```powershell
.\build-highs\bin\TSP_XGBOOST.exe --help
```

### 11.4 Command Prompt (cmd.exe) environment syntax

The `$env:NAME = ...` syntax is PowerShell syntax. If using `cmd.exe`, use:

```bat
set "XGBOOST_LIB=C:\actual\path\xgboost.dll"
set "PATH=C:\Program Files\CodeBlocks\MinGW\bin;%PATH%"
```

Then run the same `.exe` with Windows path syntax. These settings apply only
to that Command Prompt window.

## 12. Input assumptions, limitations, and troubleshooting

### Supported graph inputs

The parser handles symmetric TSPLIB coordinate types `EUC_2D`, `CEIL_2D`,
`ATT`, `MAX_2D`, and `GEO`, as well as supported symmetric explicit edge
weight matrices. It checks symmetry for `FULL_MATRIX`. Asymmetric TSP,
sequential ordering problems, precedence constraints, and unsupported
TSPLIB formats are outside the current implementation.

Tour files must contain a permutation of city IDs `1..n`. A closing repeated
city is accepted. `-1` and `EOF` terminators are supported. Training data
must have one recognized tour file adjacent to each `.tsp` file.

### Model and statistical limitations

- The classifier predicts an edge label; it does not itself construct or
  prove a tour.
- High edge recall is not the same as a high probability of retaining every
  edge in the full optimal tour.
- The best sampled tour is guaranteed to survive reduction, but it can be
  much worse than optimum.
- The model learned from one instance category may not generalize to
  different geometric distributions, dimensions, or TSP variants.
- A single random seed/run is not enough to estimate variability of this
  sampling-based method.
- The current stored train/validation/test metrics are edge-level metrics
  on a single category, not solver-level comparisons.
- The included file pairs and saved model are experiment data/artifacts; if
  the model is regenerated, keep its metadata file with it.

### Common XGBoost load error

If the program prints:

```text
error: Cannot load XGBoost shared library: C:\path\to\xgboost.dll
```

the placeholder string was passed literally. Locate the real DLL using
Section 11.1, verify `Test-Path $env:XGBOOST_LIB` returns `True`, and retry
from the same shell session. A real DLL can still fail if it is the wrong
architecture or if one of its dependent runtime DLLs is missing; use the
matching x64 library and ensure its runtime dependencies are available.

If CMake fails while fetching HiGHS, verify Git and network access and
reconfigure. If solver status is a time-limit status rather than `Optimal`,
the run may still have a valid tour; check `solver_has_solution`, objective,
bound, and gap in `summary.txt` instead of assuming optimality.

## 13. Research citation

Sun, Y., Ernst, A., Li, X., and Weiner, J. “Generalization of Machine Learning
for Problem Reduction: A Case Study on Travelling Salesman Problems.”
*OR Spectrum*, 43, 607–633 (2021).
[Article DOI](https://doi.org/10.1007/s00291-020-00604-x) ·
[arXiv preprint](https://arxiv.org/abs/2005.05847)
