# 3D HCR ROI-quality classifier (capsule)

Reproducible-run CodeOcean capsule that scores each cell in a 3-D HCR
segmentation for **quality** — `good`, `bad_ok`, `bad`, `merged` — and writes the
per-cell class-probability **contract** consumed by the coregistration matcher.

It wraps the [`mfish-roi-classifier`](https://github.com/jkim0731/mfish-roi-classifier)
package (installed in `environment/postInstall`). The model is **self-contained**:
**100 µm features, 405-only** (no upstream model, no other channels).

## Reproducible Run

Clicking **Reproducible Run** executes `code/run` → `code/run_capsule.py` for one
subject (`--subject_id`, set via the app panel):

```
build-features  (tight-bbox + unified single-pass extraction)
      ↓
predict         (LightGBM binary keep + 4-class)
```

### Parameters (`.codeocean/app-panel.json`)
| param | meaning |
|---|---|
| `subject_id` | HCR subject id, e.g. `790322` (**required**) |
| `feat_workers` | feature-extraction worker processes; `0` = auto (cpu−2) |

### Inputs (attach as data assets, mounted under `/root/capsule/data`)
The subject is resolved by glob:
- `{subject_id}*ctl-czstack-hcr-coreg_*`  (coreg dir)
- `HCR_{subject_id}_*_processed_*`        (HCR processed dir: segmentation + 405)

### Model
`run_capsule.py` reads the model from `--models_dir` (default
`/mfish-roi-classifier/models`, the version vendored in the installed package).
To use a different model, attach a model data asset and pass its path.

### Outputs (to `/root/capsule/results`)
- `{subject_id}_features_all.parquet` — per-cell feature matrix (100 features)
- `{subject_id}_roi_quality_proba.parquet` — **contract**: `hcr_id, p_bad, p_bad_ok,
  p_good, p_merged, human_label`

## Notes
- Regenerable caches (tight-bbox) go to `/root/capsule/scratch`, not the output asset.
- Part of a 3-capsule workflow: this (extract + infer) →
  `capsule-3D-HCR-ROI-labeling` (interactive labeling) →
  `capsule-3D-HCR-ROI-classifier-training` (MLflow-tracked retraining).
