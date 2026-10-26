# Gait IMU data-science pipeline

Exploratory analysis of ESP32 + MPU6050 (GY-521) gait recordings in `Data/` (read-only source CSVs).

## Run

```
pip install -r requirements.txt
python -m src.pipeline                       # validate, eda, features, stats, explore, models, report
python -m src.pipeline --stages validate features stats report
python -m pytest -q
```

All tunables live in `configs/default.yaml` (window length/overlap, filters, statistics, model options, seed).

## Layout

| path | purpose |
|---|---|
| `Data/` | original CSVs (never written to; SHA-256 checked before/after each run) |
| `processed_data/` | feature tables (`features_windows.csv`, `features_recordings.csv`), `feature_config.json` |
| `src/data_validation/` | discovery, filename parsing, pairing, data-quality checks |
| `src/signal_processing/` | regularisation, gravity estimate, filters, magnitudes, windowing |
| `src/features/` | time/magnitude/frequency/cross-channel features |
| `src/statistics/` | paired left-right tests; correlation, PCA, variability |
| `src/models/` | anomaly detection, clustering, grouped left/right classification |
| `src/visualization/`, `src/eda/` | plots and EDA runner |
| `src/report.py` | writes `reports/final_report.md` from the generated tables |
| `reports/` | `final_report.md`, `data_quality/`, `tables/`, `models/`, `figures/`, `processing_log.json`, `run_summary.json` |
| `tests/` | unit + integration tests |

Notes: on Windows `data/` and `Data/` are the same folder, so derived data goes to `processed_data/`; the pipeline refuses to write inside the raw data directory.

## Scientific audit

`python -m src.pipeline` also runs the audit stage (`src/audit/`), writing tables to `reports/audit/`. The narrative audit is `reports/scientific_audit.md`. A pre-audit copy of all outputs is kept in `audit_baseline/` (git-ignored).
