# Environment and runtimes

This file records the machine and the software stack used for every number in
the paper and in `results/`.

## Hardware and operating system

| Item | Value |
| --- | --- |
| CPU | Intel Core i9-14900HX, 24 cores / 32 logical processors |
| Memory | 31.8 GB |
| GPU | none; all runs are CPU-only |
| OS | Windows 11 (build 10.0.26100) |

No CUDA device is present, and `torch.cuda.is_available()` is `False`. Every
model in this paper is small enough that the CPU path is sufficient; the
largest single cost is the k-nearest-neighbour graph over 257,673 records.

## Software

| Package | Version |
| --- | --- |
| Python | 3.12.10 |
| PyTorch | 2.15.0+cpu |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 |

Installation:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions in the table above; `torch` is the CPU
build, so no CUDA toolkit is needed.

`torch_geometric` is **not** used. Graph propagation is implemented with
`torch.sparse` tensors and the spectral step with `scipy.sparse.linalg`, which
keeps the dependency list short and lets the code handle the full 257,673-node
graph in memory.

## Data and splits

| File | Rows |
| --- | --- |
| `data/unsw_full/UNSW_NB15_training-set.csv` | 175,341 |
| `data/unsw_full/UNSW_NB15_testing-set.csv` | 82,332 |

Two protocols are used. Protocol A is the official file split: an 80/20
stratified partition of the training file gives 140,272 training and 35,069
validation records, and the test file provides 82,332 test records. Protocol B
is a stratified 80/10/10 partition of the 257,673 pooled records, drawn with
`train_test_split(random_state=42)` and giving 206,138 training, 25,767
validation and 25,768 test records. Features are the 39 numeric columns plus
one-hot encodings of `proto`, `service` and `state`, 194 dimensions in total;
the scaler is fitted on the training partition only.

## Seeds

The reported multi-seed results use seeds 1, 2, 3, 4 and 5 for every model. The
descriptive single-run tables use seed 42. Seeds are passed to
`torch.manual_seed`, `numpy.random.default_rng` and to the scikit-learn
estimators; the kNN graph, the Jordan decomposition and the two splits are
deterministic given the data, so the seed only affects initialisation, dropout
and the random draws of the random-forest head.

## Measured runtimes (protocol A, seed 42)

| Step | Time |
| --- | --- |
| kNN graph, 257,673 nodes, 575,541 edges | 56--140 s |
| Jordan decomposition (26 eigenpairs) | 2 s |
| JM (eigen-only branch) | 36 s |
| SVM | 9 s |
| LSTM | 172 s |
| Random-Forest-GNN | 278 s |
| JM-GNN | 208 s |
| JM-LSTM | 61 s |
| JM-LLM (three closed-loop rounds) | 4 s |

One complete seven-model cache therefore costs about 13 minutes per seed,
dominated by Random-Forest-GNN and JM-GNN.

Additional experiments:

| Experiment | Time |
| --- | --- |
| `e1` per-class and per-category metrics | seconds (reads the cache) |
| `e3` attention ablation, one seed | about 60 s after the graph is built |
| `e4` directed spectrum | about 6 min (dense eigendecomposition of a 1,500-node replica) |
| `e5` controls, five seeds | about 20 min per protocol (seed 1 runs the full 120 epochs of the random-graph GNN; later seeds early-stop) |
| `e6` false-positive cost | seconds (reads the cache) |
| `e6-adv` evasion, both models | about 8 min |
| `e7` XGBoost and LightGBM, five seeds | about 4 min per protocol (3000-tree cap, early stopping at 419–1152 trees) |
| `e8` prototype sources, five seeds | about 20 min (graph 30 s, k-means 3 s, GMM 14 s, then 20 training runs) |

Two notes on reproducibility. First, the truncated eigen-only branch is
sensitive to the starting vector of ARPACK: repeated runs of `e4` vary between
0.47 and 0.49 accuracy at a tuned threshold, and the fixed-threshold value used
in Table 1 is stable at 0.44. The same sensitivity moves the Jordan-prototype
reference of `e8` by about 0.24 points: for seed 1 it gives 0.9028, 0.9004 and
0.9019 test accuracy in three independent decompositions. The four variants of
`e8` therefore share one decomposition, which keeps their paired differences
valid, but the absolute Jordan row moves within that band across runs. Second, `eigs` does not converge on the full
257,673-node directed matrix, which is why the spectrum is computed on a
1,500-node replica of the same construction; this is stated in the caption of
the directed-spectrum table in Section 5.

## What is pinned

`requirements.txt` pins the package versions listed above, and the splits,
the graph and the spectral step are deterministic given the data and
`random_state=42`, so a rerun on the same stack reproduces the tables. There is
no container image: the operating system, CPU model and memory are recorded
here instead. A different stack may move the last digit of the accuracies.
