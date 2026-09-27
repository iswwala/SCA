# Project Structure Guide

This repository is organized for side-channel analysis research rather than as a single monolithic script collection.

## Main Areas

| Area | Purpose |
| --- | --- |
| `baselines/` | Third-party methods, official implementations, and reproduction-only code. Keep these close to their original form. |
| `src/` | Our maintained implementation. New architectures and reusable code should live here. |
| `experiments/` | Experiment-specific scripts, configs, ablations, and historical trials. |
| `data/` | Local datasets and traces. This can stay outside WSL and be linked in. |
| `outputs/` | Generated figures, logs, GE curves, trained models, and comparison outputs. Use descriptive experiment names such as `framework-baseline-comparison`, `cdan-sca-baselines`, and `cdan-sca-evaluation`. |
| `references/` | Papers and source materials. |
| `docs/` | Research notes, design decisions, literature comparison, and experiment records. |

For the maintained workflow, start at `experiments/README.md`. Each active experiment has its own `README.md`, `configs/`, and `scripts/`; generated files are kept under the matching `outputs/results/<experiment>/` directory.

## Baseline Separation

Baseline code should be treated as reference material:

- Do not heavily modify baseline code directly.
- If a baseline needs adaptation, create a wrapper or copied experiment under `experiments/`.
- Record baseline assumptions and metrics in `docs/literature/` or the experiment README.

Current baseline groups:

- `baselines/ascad/ASCAD_OFFICAL/`
- `baselines/cdpa/CDPA/`
- `baselines/official_scripts/baseline_official/`

## Recommended Experiment Template

```text
experiments/<date-or-name>/
  README.md
  configs/
  scripts/
```

`experiments/legacy/` is intentionally excluded from the active workflow. Existing baseline and legacy paths are retained for reproducibility; new code should not be added there.

The experiment README should record:

- Source device and target device
- Dataset and trace window
- Leakage model and labels
- Network architecture
- Domain adaptation strategy
- Metrics, especially GE/SR/accuracy
- Main conclusion

## New Architecture Work

For the new AES cross-device structure:

- Put reusable modules in `src/framework/models/`.
- Put training logic in `src/framework/trainers/`.
- Put experiment configs in `src/framework/configs/` first.
- Once an experiment becomes large, mirror it under `experiments/<name>/`.

This keeps the research framework clean while still allowing exploratory trials.
