# SAFER-CBI calculation source

This source-only capsule contains preprocessing, modeling, and aggregate numeric
calculation APIs. Protected inputs, trained weights, predictions, checkpoints, and
generated results are not included.

## Layout

- `safer_cbi/contracts.py`: canonical feature, category, output, model, and horizon contracts
- `safer_cbi/preprocessing/`: in-memory preprocessing calculations
- `safer_cbi/modeling/`: model architectures, training, refitting, and evaluation
- `safer_cbi/analysis/`: aggregate numeric analysis and the R mixed-effects model
- `config/`: result-free model search settings
- `environment-R.yml` and `tools/`: the pinned R environment and package installer

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The single `requirements.txt` covers preprocessing, modeling, and aggregate analysis.
The package does not search for repository files. Callers supply canonical frames,
site references, thresholds, predictions, demographics, and attribution
payloads explicitly. Study-specific inputs and historical-format adapters are not
bundled here.

The mixed-effects calculation requires the included R environment:

```bash
conda env create -f environment-R.yml
conda run --no-capture-output -n R Rscript tools/install_r_packages.R
```

Released under the MIT License. See `LICENSE`.
