# Bengal Election 2026 Forecast Studio

A constituency-level forecasting and scenario analysis project for the West Bengal Assembly election. The repository now includes a cleaner training pipeline, a safer inference layer, and a professional Streamlit dashboard focused on reliability instead of decorative output.

## What is included

- A rebuilt Streamlit application in `app.py`
- A reusable modeling package in `bengal_election/`
- A root training entrypoint in `train_model.py`
- Historical election data in `data/raw/West_Bengal_AE.csv`
- Backward compatibility with the legacy `models/bengal_rf_model.pkl` and `models/encoders.pkl` artifacts

## Model improvements

The original project used a very thin model based mostly on encoded party names, constituency labels, vote share, and a rough age estimate. The new pipeline is designed to be more dependable:

- Cleans and normalizes party labels such as `CPM` to `CPIM` and `FBL` to `AIFB`
- Builds lag features for each constituency-party combination
- Tracks previous winner information for each constituency
- Uses turnout, contest intensity, candidate continuity, and prior-cycle signals
- Compares multiple tree-based models and keeps the strongest validation result
- Saves one consolidated bundle with metrics and feature importance metadata

The training bundle is written to `models/bengal_election_bundle.pkl`.

## Dashboard improvements

The Streamlit app has been rebuilt around four practical views:

- `Scenario Lab` for controlled vote-share swings and win-probability comparisons
- `Constituency Profile` for vote share and winning margin history
- `Model Reliability` for held-out validation metrics and model selection results
- `Data Table` for direct inspection of election rows

The interface avoids emojis, uses a more restrained visual system, and includes clearer failure handling when artifacts are missing or outdated.

## Project structure

```text
Bengal-Election-2026/
├── app.py
├── train_model.py
├── requirements.txt
├── bengal_election/
│   ├── __init__.py
│   └── pipeline.py
├── data/
│   ├── raw/West_Bengal_AE.csv
│   └── processed/adr_parsed_raw.csv
├── models/
│   ├── bengal_rf_model.pkl
│   ├── encoders.pkl
│   └── bengal_election_bundle.pkl   # created after retraining
└── pipeline_archive/
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the improved bundle

```bash
python train_model.py
```

This creates `models/bengal_election_bundle.pkl` and prints validation metrics.

If the new bundle is not present, the app will still attempt to use the legacy model artifacts, but the full reliability view will only be available after retraining.

## Run the dashboard

```bash
streamlit run app.py
```

## Notes

- The primary dataset currently available in this repository contains about 23,900 election rows.
- The archived ADR file in `data/processed/adr_parsed_raw.csv` is incomplete and is no longer treated as a core dependency for the forecast pipeline.
- `pipeline_archive/` is retained for historical reference, not for the live app path.
