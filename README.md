# JM-LLM: reproduction package

Code, results and configuration for

> **JM-LLM: Threat Prediction in Multi-vector Networks via Jordan Matrix and LLM
> Collaboration**
> Zicheng Dai, Lansheng Han, Dongjun Zhu, Xiaohu Ye
> *Journal of Information Security and Applications* (submitted)

The package reproduces every number in the manuscript on a single CPU
workstation. It is a pure-PyTorch implementation: it does not depend on
`torch_geometric`, and the graph, the sparse spectral step and the seven models
all run on the full 257,673-node UNSW-NB15 corpus without a GPU.

## What the paper claims

The framework has two branches. An algebraic branch decomposes the propagation
matrix, retains the 26 largest-modulus eigenpairs and predicts from the
resulting modes; a semantic branch is a 128-64-2 feed-forward encoder over the
194-dimensional flow features. The branches are fused by an equal-weight average
of their probabilities with a validation-tuned threshold, and a mode-guided
attention module can modulate the encoder with mode prototypes.

The experiments report three findings, and the negative ones are reported as
such:

1. UNSW-NB15 is heavily duplicated. 104,113 of its 257,673 records repeat an
   exact feature vector (40.4%), and 45.5% of the test rows of a naive random
   split repeat a training vector. Protocol C therefore deduplicates first:
   153,146 unique flows, stratified 80/10/10, zero test/train overlap.
2. On protocol C JM-LLM is the best of the seven models at 0.9011 ± 0.0011 test
   accuracy and 0.8997 ± 0.0012 macro-F1 with the lowest false-positive rate, but
   its margin over a plain LSTM is 0.39 ± 0.51 points and five seeds do not
   resolve it. The same encoder trained alone reaches 0.8963 ± 0.0066, i.e. the
   fusion is inert under this protocol.
3. The neighbourhood graph does carry information: permuting the target
   endpoints of the edges, while keeping out-degrees and weights, costs the
   graph model 16.9 points under protocol A, 22.2 under B and 18.6 under C.

## Requirements

Python 3.12, CPU only. The versions that produced the results are pinned:

```bash
pip install -r requirements.txt
```

## Data

Download the two release files of
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place
them here:

```
data/unsw_full/UNSW_NB15_training-set.csv
data/unsw_full/UNSW_NB15_testing-set.csv
```

`data/` is git-ignored: the benchmark is redistributed by its authors, not by
this repository. A different location can be given with the environment
variable `UNSW_NB15_DIR`.

The two files are pooled into one graph of 257,673 nodes. Each flow is a node;
edges come from a k-nearest-neighbour graph (k=3) with Gaussian-kernel weights
over the standardised feature space. Features are the 39 numeric columns plus
one-hot encodings of `proto`, `service` and `state`, 194 dimensions in total.
The scaler is fitted on the training partition only.

Two protocols are used throughout:

| Protocol | Split | Sizes (train / validation / test) |
| --- | --- | --- |
| A | official file split, 20% of the training file held out for validation | 140,272 / 35,069 / 82,332 |
| B | stratified 80/10/10 over the pooled records, `random_state=42` | 206,138 / 25,767 / 25,768 |
| C | exact-duplicate removal, then stratified 80/10/10, `random_state=42` | 122,516 / 15,315 / 15,315 |

Protocol C is the primary comparison. Records are grouped by their exact
194-dimensional feature vector, one record per group is kept, and a group is
dropped entirely when its members carry conflicting labels (414 groups, 1,758
records). The result is 153,146 flows with an attack share of 44.4%, against
63.9% in the pooled corpus. Protocols B and A are retained as controls: 45.53% of
the protocol-B test rows and 9.81% of the protocol-A test rows have an exact
feature vector in their own training partition, against 0% for C. Because C also
changes the class prior, cross-protocol comparisons are made on macro-F1 rather
than on accuracy.

## Layout

| Path | Contents |
| --- | --- |
| `common.py` | Data loading and cleaning, kNN graph, sparse spectral decomposition, propagation score |
| `models.py` | GCN and LSTM backbones, `LMModel` encoder, `CollaborativeModel` fusion, batched prediction |
| `correction.py` | Decision-layer correction (temperature scaling, validation threshold, EM/BBSE prior estimation) |
| `normal_profile.py` | Normal-only autoencoder branch (optional, used in the earlier experiments) |
| `run_full_comparison.py` | The seven-model comparison of Tables 1 and 2 |
| `run_n_sweep.py` | The 40-point sweep over the retained mode count, Fig. 4 |
| `experiments.py` | The five-seed experiments and controls: `cache`, `e1`–`e6`, `e6-adv` |
| `results/` | Every CSV/JSON behind the tables and figures of the manuscript |
| `figures/` | The mode-count and eigenvalue figures, as they appear in the paper |
| `ENVIRONMENT.md` | Machine, package versions, measured runtimes, known numerical caveats |

## Running

```bash
# Table 1 and Table 2
python run_full_comparison.py
python run_full_comparison.py --protocol random

# Fig. 4: 40-point mode sweep (about 2 h on 24 threads)
python run_n_sweep.py

# Cache the seven models for one seed, then run any experiment
python experiments.py dedup                       # protocol C statistics
python experiments.py cache --protocol dedup --seed 1
python experiments.py e1    --protocol official --seed 1
python experiments.py e2    --protocol dedup  --seeds 1 2 3 4 5
python experiments.py e3    --protocol dedup  --seeds 1 2 3 4 5
python experiments.py e4
python experiments.py e5    --protocol dedup  --seeds 1 2 3 4 5
python experiments.py e6    --protocol dedup  --seeds 1 2 3 4 5
python experiments.py e6-adv --protocol official --seed 42
```

`cache` writes `experiments/cache_{protocol}_seed{n}.npz` and one
`part_{protocol}_{seed}_{model}.npz` per model, so an interrupted run resumes at
the model level. Every other command reads the cache and writes its own CSV or
JSON into `experiments/`. The `results/` directory of this repository contains
the outputs of those commands, so each published number can be traced without
retraining.

## What each experiment does

| Command | Output | Purpose |
| --- | --- | --- |
| `dedup` | `dedup_stats.json` | Duplicate-group statistics and the test/train duplicate coverage of each protocol |
| `e1` | `e1_per_class_*.csv`, `e1_attack_categories_*.csv`, `e1_confusion_*.csv` | Per-class precision/recall/F1/FPR/FNR and per-attack-category recall with support |
| `e2` | `e2_per_seed_*.csv`, `e2_mean_std_*.csv`, `e2_paired_*.csv` | Five-seed means, standard deviations and paired t tests |
| `e3` | `e3_attention_*.csv`, `e3_attention_*.json` | Mode-prototype attention on/off, paired over seeds, with parameter counts |
| `e4` | `e4_directed_spectrum.json` | Non-normality, spectrum and defectiveness of the directed propagation matrix; directed diffusion versus the symmetrised 26-mode branch |
| `e5` | `e5_controls_*.csv`, `e5_paired_*.csv` | Rewired-graph control and the encoder trained alone at 33,346 and 62,506 parameters |
| `e6` | `e6_fpr_cost_*.csv` | False-positive/false-negative rates and cost per 10^4 flows at FP:FN ratios 1:1, 1:5, 1:10 |
| `e6-adv` | `e6_adversarial_*.csv` | White-box one-step FGSM evasion with a random-perturbation control |

## Seeds and hardware

Multi-seed results use seeds 1–5; the descriptive single-run tables use seed 42.
The graph, the splits and the spectral decomposition are deterministic given the
data. All runs are CPU-only on an Intel Core i9-14900HX (24 cores / 32 threads)
with 31.8 GB of RAM. Measured runtimes, the package versions and the numerical
caveats (ARPACK restarts) are in `ENVIRONMENT.md`.

## Deviations from a literal reading of the paper

1. The semantic branch is a 128-64-2 encoder trained from scratch, not a
   pretrained language model. The paper uses "LLM" as an architectural label and
   discloses the limitation in Section 3.1 and Section 5.15.
2. The Jordan form is computed as a truncated diagonal decomposition of the
   symmetrised adjacency, so blocks of order greater than one are not
   represented. `e4` shows that the directed matrix has no defective eigenvalue
   on this graph, so such blocks would not appear even if they were
   representable.
3. The reinforcement-policy-analysis quantity of the reference implementation is
   replaced by accuracy and macro-F1, because the original definition
   `TP/(TP+FP+FN)` is not comparable across class priors.
4. The two branches are fused by an equal-weight average of probabilities with a
   validation-tuned decision threshold; the structural correction term is
   inactive (λ=0) in every reported run.
5. Protocol C exists because 40.4% of the corpus consists of exact duplicate
   rows, and protocol B leaves 45.5% of its test rows open to duplicate leakage.
   Protocol A additionally differs in class prior and in generating environment;
   the manuscript quantifies both effects in Section 5.14.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper; if you use the code, cite this
repository as well.
