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

1. Under the random stratified split (protocol B) JM-LLM is the best of the seven
   models in all five seeds, at 0.9343 ± 0.0013 test accuracy, but the same
   encoder trained alone, with no graph and no Jordan branch, reaches
   0.9337 ± 0.0015 — the difference is not resolved.
2. Under the official file split (protocol A) the encoder alone is 2.03 points
   *better* than the fusion, because the algebraic branch is itself at 0.44
   accuracy and enters the average with weight one half.
3. The neighbourhood graph does carry information: permuting the target
   endpoints of the edges, while keeping out-degrees and weights, costs the
   graph model 16.9 points under protocol A and 22.2 under protocol B.

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

Protocol B is the primary comparison, because the two official files are not a
random split of one capture (the manuscript quantifies this in Section 5.14).

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

# Cache the seven models for one seed, then run any review experiment
python experiments.py cache --protocol official --seed 1
python experiments.py e1    --protocol official --seed 1
python experiments.py e2    --protocol official --seeds 1 2 3 4 5
python experiments.py e3    --protocol official --seeds 1 2 3 4 5
python experiments.py e4
python experiments.py e5    --protocol official --seeds 1 2 3 4 5
python experiments.py e6    --protocol official --seeds 1 2 3 4 5
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
5. Protocol B exists because the two release files differ in class prior and in
   generating environment; Section 5.14 of the manuscript quantifies both.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper; if you use the code, cite this
repository as well.
