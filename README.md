# Risk-Constrained Sparse Refinement from Aggregate Feedback

Lightweight code release for experiments on risk-constrained sparse refinement
from non-identifying aggregate feedback in pediatric gait multi-label
assessment.

This repository is intentionally code-focused. It does not include raw
pediatric gait videos, raw pose-frame directories, original label files, full
GaitRec downloads, manuscript build files, or submission-package logs.

## Repository Contents

- `src/`: core Python scripts and local utility modules.
- `results_summary/`: compact derived summaries used to check the main
  numerical claims.
- `README.md`, `requirements.txt`, `.gitignore`: minimal GitHub project files.
- `PACKAGE_MANIFEST.md/csv`: file list with sizes and SHA256 checksums for this
  lightweight package.

## Main Scripts

- `run_offline_feedback_simulation.py`: single-bit aggregate-feedback simulator.
- `run_offline_sparse_state_posterior.py`: sparse-state posterior variant.
- `run_offline_external_adaptation_baselines.py`: threshold and LLP-style
  external baselines.
- `analyze_patient_cluster_bootstrap.py`: patient-clustered bootstrap summaries.
- `analyze_budget_matched_oracle_skyline.py`: budget-matched oracle headroom.
- `run_initial_predictor_robustness.py`: initial-predictor robustness checks.
- `run_synthetic_structured_feedback_task.py`: synthetic boundary check.
- `run_gaitrec_external_validation.py`: GaitRec boundary check.
- `make_paper_figures_polished.py` and `make_gate_interpretability_figure.py`:
  figure-generation scripts.

Feature/model utility scripts such as `pose_feature_baseline.py`,
`physics_gait_features.py`, `gait_event_phase_features.py`, and
`side_canonical_event_model.py` are included because the main simulator imports
them.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The original pediatric gait dataset is not redistributed. The GaitRec dataset
is also not bundled because it should be downloaded from its public source by
the user. Scripts that require raw data expect local paths supplied through
their command-line arguments or the path constants documented inside each
script.

The included `results_summary/` files are derived aggregate summaries. They are
provided so readers can inspect the numerical evidence without access to
restricted raw data.

## Responsible Use

This code is for research and reproducibility. It generates auxiliary gait
assessment and refinement analyses and is not a diagnostic or treatment system.
Hidden-label outcomes and harmful-update indicators are offline audit
diagnostics only; they are not available to the refinement algorithm during
deployment.

