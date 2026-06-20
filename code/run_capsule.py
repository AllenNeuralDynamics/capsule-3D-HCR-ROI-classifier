"""Reproducible-run entry for the 3-D HCR ROI-quality classifier capsule.

For one subject it (1) builds the per-cell feature matrix — tight-bbox + the
unified single-pass extractor (µm shape / axis / surface / protrusion / 405
intensity / adjacency / neighbour-quality, 405-only, self-contained) — and
(2) runs the classifier, writing to the output asset:

    {sid}_features_all.parquet         per-cell feature matrix (100 features)
    {sid}_roi_quality_proba.parquet    contract: hcr_id, p_bad, p_bad_ok, p_good, p_merged

Human labels are NOT part of this contract — labeling is a separate, later step.

Data resolution (mfish-roi-classifier): the subject's HCR-processed asset is globbed
under MFISH_DATA_ROOT (= --input_dir) as ``HCR_{sid}_*_processed_*`` — no coreg dir is
needed (the 405-only classifier uses HCR data only).

Model: the CodeOcean **model data asset attached under --input_dir** is the source of
truth. It is auto-detected by FORMAT at ANY depth — the directory that actually contains
``roi_quality_4class.txt`` + ``roi_quality_meta.json`` (found via rglob), so the
MLflow-on-CodeOcean ``{asset}/{name}/model/artifacts/`` layout works without assuming the
files sit at the top of the asset. Override with --models_dir.

Build + predict run **in-process** (direct library calls), not via the CLI subprocess.
``--max_cells N`` runs a quick smoke test on only the N lowest-z ROIs (a narrow z-band →
few strips), using an isolated temporary cache so the production caches are untouched.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path


def _find_model_dir(data_root: Path) -> Path:
    """Locate the model by FORMAT: the directory that actually contains BOTH
    roi_quality_4class.txt and roi_quality_meta.json, at any depth (rglob) — handles a
    plain model asset or the MLflow-on-CodeOcean ``{asset}/{name}/model/artifacts/`` layout.
    Returns that exact directory so MFISH_MODELS_DIR points right at the files. The large
    HCR_* data asset is skipped so rglob never walks the segmentation zarr."""
    for top in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if top.name.startswith("HCR_"):
            continue   # the terabyte-scale HCR data asset is never the model
        meta = next((m for m in top.rglob("roi_quality_meta.json")
                     if (m.parent / "roi_quality_4class.txt").exists()), None)
        if meta is not None:
            return meta.parent
    raise FileNotFoundError(
        f"no ROI-quality model found under {data_root} — expected a directory (at any depth) "
        f"with roi_quality_4class.txt + roi_quality_meta.json (attach the model data asset).")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HCR ROI-quality: extract per-cell features + predict for one subject.")
    ap.add_argument("--subject_id", required=True, help="HCR subject id, e.g. 790322")
    ap.add_argument("--input_dir", default="/root/capsule/data",
                    help="Mounted data root holding the subject's HCR-processed asset + model asset.")
    ap.add_argument("--output_dir", default="/root/capsule/results",
                    help="Output asset directory for the feature matrix + proba contract.")
    ap.add_argument("--models_dir", default="",
                    help="Directory with roi_quality_4class.txt + roi_quality_meta.json. "
                         "Default: auto-detect the attached model data asset under --input_dir "
                         "by format (any depth). Set to override.")
    ap.add_argument("--feat_workers", type=int, default=0,
                    help="Feature-extraction worker processes (0 = package default, cpu-2).")
    ap.add_argument("--max_cells", type=int, default=0,
                    help="SMOKE TEST: if >0, extract features for only the N lowest-z ROIs "
                         "(a narrow z-band → few strips → fast); uses an isolated temp cache.")
    args = ap.parse_args()

    sid = str(args.subject_id).strip()
    if not sid:
        raise SystemExit("subject_id is required (e.g. --subject_id 790322)")
    limit = max(0, int(args.max_cells))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # Smoke test writes a subset tight-bbox cache → isolate it so it can't clobber the
    # production full-subject cache (which a later full run would otherwise reuse).
    cache = Path(tempfile.mkdtemp(prefix="mfish_smoke_")) if limit \
        else Path("/root/capsule/scratch/mfish_cache")
    cache.mkdir(parents=True, exist_ok=True)

    models_dir = args.models_dir or str(_find_model_dir(Path(args.input_dir)))
    print(f"[capsule] model dir: {models_dir}", flush=True)

    # Configure the package via env BEFORE importing it: config.py reads these at import,
    # and model.py loads FEATURE_COLUMNS from the model meta at import. (os.environ is
    # inherited by feat_shape's multiprocessing workers.)
    os.environ["MFISH_DATA_ROOT"] = args.input_dir
    os.environ["MFISH_ROI_QUALITY_DIR"] = str(out)   # features_all + proba contract -> output asset
    os.environ["MFISH_CACHE_DIR"] = str(cache)       # tight-bbox cache (scratch, or temp for smoke)
    os.environ["MFISH_MODELS_DIR"] = models_dir
    if args.feat_workers > 0:
        os.environ["MFISH_FEAT_WORKERS"] = str(args.feat_workers)

    # ── in-process: build features, then predict (no CLI subprocess) ─────────────
    from roi_classifier import config as cfg
    from roi_classifier import feat_shape
    from roi_classifier.benchmark_data_loader import load_subject
    from roi_classifier.feat_tight_bbox import build_tight_bbox
    from roi_classifier.features import extract_features
    from roi_classifier.model import predict

    print(f"[build-features] subject={sid}" + (f"  [SMOKE TEST: {limit} ROIs]" if limit else ""), flush=True)
    s = load_subject(sid)
    if limit:
        # keep only the N lowest-z ROIs (contiguous z → few strips → fast)
        s.hcr_centroids = s.hcr_centroids.nsmallest(limit, "z_px").reset_index(drop=True)
    tb = build_tight_bbox(s, cache=True, force=bool(limit))   # rows = cells WITH seg voxels
    found_ids = set(int(h) for h in tb["hcr_id"])
    feat_shape.compute(s, cache=True)                    # -> {sid}_features_all.parquet in out
    feat = extract_features(sid)                          # reads the cached feature parquet

    # Drop ROIs with no segmentation voxels: those centroids are absent from the orig-res
    # mask, so their feature rows are all-NaN and they can't be coregistered. Keep only the
    # cells that have voxels (present in the tight-bbox result).
    n_all = len(feat)
    feat = feat[feat["hcr_id"].astype(int).isin(found_ids)].reset_index(drop=True)
    if len(feat) < n_all:
        print(f"  dropped {n_all - len(feat)} ROI(s) with no segmentation voxels "
              f"→ {len(feat)} scored", flush=True)

    # ── predict -> contract parquet (hcr_id, p_bad, p_bad_ok, p_good, p_merged) ──
    print(f"[predict] subject={sid}  (model dir: {cfg.MODELS_DIR})", flush=True)
    _binary_score, four_class_proba = predict(feat)
    contract = four_class_proba[["hcr_id", "bad", "bad_ok", "good", "merged"]].rename(
        columns={"bad": "p_bad", "bad_ok": "p_bad_ok", "good": "p_good", "merged": "p_merged"})

    proba_path = out / f"{sid}_roi_quality_proba.parquet"
    contract.to_parquet(proba_path, index=False)
    print(f"  contract -> {proba_path}  shape={contract.shape}  cols={list(contract.columns)}", flush=True)

    print(f"[capsule] subject {sid} done. Outputs in {out}:", flush=True)
    for f in sorted(out.glob(f"{sid}_*.parquet")):
        print("    ", f.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
