# Cross-Project Reopened Bug Prediction

**An empirical study of transferability, semantic embeddings, and data-quality confounds in reopened-bug prediction.**

MSc Data Analytics — Research Practicum · National College of Ireland

---

## Overview

Reopened bugs are bugs that are closed and later re-opened, causing costly rework for development teams. Predicting which bugs are likely to be reopened could help teams avoid that rework. Prior work (Tagra et al., 2022) built per-project prediction models across 47 open-source projects but noted that small, data-scarce projects are hard to model on their own data.

This project asks a natural follow-up: **can we predict reopened bugs for a project by learning from *other* projects (cross-project training), using modern semantic text embeddings — especially to help the small projects?**

The short answer, after a thorough and honest evaluation, is **no** — reopened-bug patterns are strongly project-specific and do not transfer. This repository documents the full study, including the faithful baseline reproduction, the cross-project experiments, a mixed-project experiment, and a data-quality analysis that surfaced confounds not reported in the original paper.

---

## Research questions

**Main RQ:** *To what extent do reopened-bug prediction models trained on multiple source projects transfer to unseen target projects, and what factors limit their transferability?*

- **SQ1** — How does a cross-project (leave-one-project-out) model compare to a per-project baseline across projects of different sizes?
- **SQ2** — Do semantic text embeddings (Sentence-BERT) improve cross-project transfer over project-local text features (TF-IDF)?
- **SQ3** — Does augmenting a project's own limited data with cross-project data help, especially for data-scarce projects?
- **SQ4** — What data-quality confounds affect reopened-bug datasets and distort model evaluation?

---

## Dataset

- **Source:** the 47-project JIRA dataset from Tagra et al. (2022) (replication package).
- **Size:** ~168,881 bug reports; 8,886 reopened (~5.3% — a heavily imbalanced problem).
- **Projects:** 47 Apache-ecosystem projects (ACCUMULO, AMBARI, CASSANDRA, FLEX, SPARK, …), one CSV per project.
- **Features per bug:** 12 numeric (comment counts, time-to-close, reporter/assignee experience, …), 4 categorical (component, priority, last status, time-of-day), and 2 text fields (description, comments) — all computed **before the bug's first reopening** to avoid data leakage.

> Note: this shared replication extract is ~10% smaller than the dataset reported in the paper, so two projects (BAHIR, CTAKES) fall below the ≥10-reopened threshold here.

---

## Methodology

### 1. Baseline (faithful reproduction of Tagra et al.)
- One-hot encoding (categorical) + TF-IDF → Naive Bayes (text) + numeric features
- SMOTE on training folds only (class imbalance)
- AutoSpearman correlation/redundancy feature filtering
- Random Forest classifier
- 10-fold stratified cross-validation, evaluated with AUC (threshold 0.7 = "acceptable")

### 2. Cross-project model (leave-one-project-out)
- Train on all other 46 projects, test on the held-out project (strict, zero-shot)
- Text represented with **Sentence-BERT embeddings** (`all-MiniLM-L6-v2`, 384-dim) so text can transfer across projects
- Also evaluated **numeric-only** and **embeddings-only** variants to isolate each signal

### 3. Mixed-project (augmented) experiment
- Train on a project's own data **plus** all other projects; compare to a within-only model on identical folds
- Focused on the smallest / most data-scarce projects (the hypothesis target)

### 4. Data-quality analysis
- Per-feature single-feature AUC scan across all projects to detect confounds / leakage

---

## Key findings

| Approach | Mean AUC | Projects ≥ 0.7 |
|---|---|---|
| Per-project baseline (reproduction) | **0.73** | 25 / 45 (56%) |
| Cross-project + embeddings | 0.569 | 0 / 45 |
| Cross-project numeric-only | 0.595 | 1 / 45 |

- **SQ1 — Cross-project transfer fails.** The cross-project model beat the per-project baseline on **0 of 45** projects; several scored near or below chance (e.g. GERONIMO 0.42).
- **SQ2 — Embeddings do not transfer the reopen signal.** Embeddings alone were near-chance (one project exactly 0.500), and *numeric-only (0.595) outperformed numeric+embeddings (0.569)* — the embeddings **diluted** the signal rather than adding to it.
- **SQ3 — Mixed-project augmentation does not help.** On the smallest projects (full settings), adding cross-project data gave **no reliable benefit** — 6 of 9 projects got *worse*, average delta ≈ −0.04.
- **SQ4 — Previously-unreported data confounds.** The reproduction scored higher than the paper (56% vs 34%); this was traced to 3 projects where a single feature almost perfectly separates the classes for a bookkeeping reason:
  - **FLEX** — `assignee_experience` (single-feature AUC 0.94): thousands of bugs were bulk-closed under one high-experience account and never reopened.
  - **BEAM / CASSANDRA** — `changelog_Priority_changed` (0.96 / 0.92): near-deterministic split.
  - These are **not** the data leak from the earlier 3-project studies; they are confounds *within Tagra's own dataset* that the paper does not discuss.
- **Coverage — the one genuine cross-project advantage.** Projects too small to model per-project at all (BAHIR, 5 reopened; CTAKES, 8) *can* receive a usable cross-project prediction (AUC 0.59, 0.70 — reported with a small-sample caveat).

### Why transfer fails (mechanism)
- **Project-local features** (TF-IDF, categorical) *capture* the reopen signal but **cannot transfer** across projects (different vocabularies and components).
- **Transferable features** (semantic embeddings) *do* transfer but **do not capture** the reopen signal (they encode what a project is *about*, not whether a bug will reopen).
- No feature type both captures *and* carries the signal — so reopened-bug prediction is fundamentally hard to transfer across projects.

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_data_setup.ipynb          # dataset verification, class balance, per-project summary
│   ├── 02_baseline.ipynb            # Tagra baseline reproduction + data-quality confound analysis
│   ├── 03_cross_project.ipynb       # embeddings, leave-one-project-out, numeric/embeddings variants
│   └── 04_mixed_project.ipynb       # within-only vs augmented experiment
├── data/
│   ├── joined_project_dataframe_*.csv   # 47 per-project datasets (not redistributed — see Tagra et al.)
│   └── emb_*.npy                        # cached Sentence-BERT embeddings per project
├── results/
│   ├── baseline_results.csv
│   ├── cross_project_results.csv        # cross-project + embeddings
│   ├── cross_project_numeric_results.csv
│   ├── comparison_all.csv               # baseline vs cross-project side by side
│   └── small_projects_full.csv          # mixed-project results (small projects)
└── README.md
```

> Adjust the notebook/file names above to match your actual files.

---

## How to run

### Requirements
- Python 3.10+
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `imbalanced-learn`, `sentence-transformers`

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn sentence-transformers
```

### Steps
1. Place the 47 project CSVs in `data/`.
2. Run `01_data_setup` to verify the dataset.
3. Run `02_baseline` to reproduce the per-project baseline and the confound analysis.
4. Run `03_cross_project` — this generates and caches the embeddings (`emb_*.npy`, a one-time ~CPU-heavy step), then runs the leave-one-project-out experiments.
5. Run `04_mixed_project` for the augmented experiment.

> Embeddings are cached to `.npy` files, so they are computed only once. All experiment loops save to CSV and are resumable.

---

## Limitations & future work

- **Label cleaning not possible with this data.** Tagra et al. recommend removing "bookkeeping" reopenings (field edits), but the replication dataset contains only *before-reopening* features, so the information needed to identify them is absent. This is left as future work pending fuller JIRA history.
- **Single embedding model.** Only `all-MiniLM-L6-v2` was tested; a software-engineering-specific model (e.g. CodeBERT) could be evaluated.
- **Confound sensitivity.** A baseline re-run with the confound features removed would quantify their exact contribution.
- **Other datasets.** Applying the pipeline to a task where cross-project transfer is known to work (e.g. cross-project defect prediction) would further isolate why reopened-bug transfer fails.

---

## References

- A. Tagra, H. Zhang, G. K. Rajbahadur, A. E. Hassan (2022). *Revisiting Reopened Bugs in Open Source Software Systems.* Empirical Software Engineering, 27(80).
- N. Reimers, I. Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.

---

## Author

**AR (Akshaya Ramkumar)** — MSc Data Analytics, National College of Ireland.

*This repository documents an honest empirical evaluation: a rigorous negative result, with a mechanistic explanation and a novel data-quality finding.*
