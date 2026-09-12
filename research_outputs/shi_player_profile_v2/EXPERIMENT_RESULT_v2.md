## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-22
- Verification Status: VERIFIED
- Version Label: shi_player_profile_exp_result_v2.0

## Experiment Result

- **ID**: shi-player-profile-v2
- **Type**: analysis + bootstrap simulation
- **Status**: completed
- **Command**: `<PROJECT_ROOT>\.venv-model\Scripts\python.exe <PROJECT_ROOT>\research\shi_player_profile_v2.py`
- **Working Directory**: `<PROJECT_ROOT>`
- **Hard Timeout**: 30 minutes
- **Observed Runtime**: approximately 10 seconds per run
- **Exit Code**: 0 on both runs

### Asset Readiness

| Asset | Assessment |
|---|---|
| Locked protocol | Present; v2.0-final |
| Identity mapping | 24 unique matches, 17 opponents |
| Match event tables | 24/24 present; one shared schema |
| Source event size | 26,851,419 bytes |
| Python runtime | Python 3.12.13 |
| Analysis libraries | NumPy 1.26.4; Matplotlib 3.11.1 |
| Version control | Workspace is not a Git repository |
| Reproducibility substitute | Input SHA-256 manifest, protocol lock, fixed seed, isolated output directory, exact two-run comparison |

### Data Acceptance

| Check | Locked value | Observed | Status |
|---|---:|---:|---|
| Matches | 24 | 24 | PASS |
| Match-player rows | 48 | 48 | PASS |
| Valid upper/lower events | 26,629 | 26,629 | PASS |
| Shi Yuqi events | 14,157 | 14,157 | PASS |
| Shi Yuqi classified events | 10,081 | 10,081 | PASS |
| Shi Yuqi unknown events | 4,076 | 4,076 | PASS |
| Shi Yuqi rallies | 1,923 | 1,923 | PASS |
| Shi Yuqi valid positions | 13,630 | 13,630 | PASS |
| Shi Yuqi valid landings | 6,459 | 6,459 | PASS |

Fourteen source rows whose `player` was neither `upper` nor `lower` were excluded and counted. No duplicate event conflict, rally-order conflict, or non-positive rally duration was detected.

### Output Files

- `normalized_events.csv`: 26,629 event rows.
- `match_metrics.csv`: 48 match-player rows.
- `metric_summary.csv`: 40 athlete-metric summary rows.
- `paired_comparisons.csv`: 13 paired primary/secondary metrics.
- `sensitivity_results.csv`: 54 S1–S9 checks.
- `composition_jsd.csv` and `composition_clr.csv`: compositional sensitivity artifacts.
- `analysis_summary.json`, `data_audit.json`, `input_manifest.json`, `protocol_lock.json`.
- Four publication-oriented PNG figures.
- `RESEARCH_REPORT_v2.md` and `RUN_LOG.md`.

### Anomalies Detected

- No runtime crash, timeout, output stall, or numerical-range failure.
- Several paired estimates reverse direction under pooled weighting or leave-one-match-out analysis. These are substantive sensitivity findings and are retained as `UNSTABLE`, not treated as execution errors.
