"""Reproducible-run entry for the 3-D HCR ROI-quality classifier capsule.

For one subject it (1) builds the per-cell feature matrix — tight-bbox + the
unified single-pass extractor (µm shape / axis / surface / protrusion / 405
intensity / adjacency / neighbour-quality, 405-only, self-contained) — and
(2) runs the classifier, writing to the output asset:

    {sid}_features_all.parquet         per-cell feature matrix (100 features)
    {sid}_roi_quality_proba.parquet    contract: hcr_id, p_bad, p_bad_ok,
                                        p_good, p_merged, human_label

Data resolution (mfish-roi-classifier): the subject's HCR-processed asset is globbed
under MFISH_DATA_ROOT (= --input_dir) as ``HCR_{sid}_*_processed_*`` — no coreg dir is
needed (the 405-only classifier uses HCR data only).

Model: the CodeOcean **model data asset attached under --input_dir** is the source of
truth. It is auto-detected by FORMAT — a directory holding ``roi_quality_4class.txt`` +
``roi_quality_meta.json`` — not a fixed asset name/path, so the model can be re-registered
and re-attached without changing this code. Override with --models_dir.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_model_dir(data_root: Path) -> Path:
    """Locate the attached model under `data_root` by its FORMAT — a directory holding both
    roi_quality_4class.txt and roi_quality_meta.json — rather than a fixed asset name/path.
    Checks each attached asset dir and its immediate subdirs (covers a plain model asset or
    an MLflow pyfunc ``artifacts/`` layout) without descending into large data trees such as
    the HCR segmentation zarrs."""
    def _is_model(d: Path) -> bool:
        return (d / "roi_quality_4class.txt").exists() and (d / "roi_quality_meta.json").exists()
    for top in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if _is_model(top):
            return top
        try:
            for sub in sorted(s for s in top.iterdir() if s.is_dir()):
                if _is_model(sub):
                    return sub
        except OSError:
            continue
    raise FileNotFoundError(
        f"no ROI-quality model found under {data_root} — expected a directory with "
        f"roi_quality_4class.txt + roi_quality_meta.json (attach the model data asset).")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HCR ROI-quality: extract per-cell features + predict for one subject.")
    ap.add_argument("--subject_id", required=True, help="HCR subject id, e.g. 790322")
    ap.add_argument("--input_dir", default="/root/capsule/data",
                    help="Mounted data root holding the subject's coreg + HCR-processed assets.")
    ap.add_argument("--output_dir", default="/root/capsule/results",
                    help="Output asset directory for the feature matrix + proba contract.")
    ap.add_argument("--models_dir", default="",
                    help="Directory with roi_quality_4class.txt + roi_quality_meta.json. "
                         "Default: auto-detect the attached model data asset under --input_dir "
                         "by format. Set to override.")
    ap.add_argument("--feat_workers", type=int, default=0,
                    help="Feature-extraction worker processes (0 = package default, cpu-2).")
    args = ap.parse_args()

    sid = str(args.subject_id).strip()
    if not sid:
        raise SystemExit("subject_id is required (e.g. --subject_id 790322)")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path("/root/capsule/scratch/mfish_cache"); cache.mkdir(parents=True, exist_ok=True)

    models_dir = args.models_dir or str(_find_model_dir(Path(args.input_dir)))
    print(f"[capsule] model dir: {models_dir}", flush=True)

    env = os.environ.copy()
    env["MFISH_DATA_ROOT"] = args.input_dir
    env["MFISH_ROI_QUALITY_DIR"] = str(out)    # features_all + proba contract -> output asset
    env["MFISH_CACHE_DIR"] = str(cache)        # regenerable tight-bbox cache -> scratch
    env["MFISH_MODELS_DIR"] = models_dir
    if args.feat_workers > 0:
        env["MFISH_FEAT_WORKERS"] = str(args.feat_workers)

    def run(*cmd: str) -> None:
        print("+ roi-classifier", *cmd, flush=True)
        subprocess.run([sys.executable, "-m", "roi_classifier.cli", *cmd], check=True, env=env)

    run("build-features", sid)   # tight-bbox + unified single-pass extraction
    run("predict", sid)          # inference -> {sid}_roi_quality_proba.parquet

    print(f"[capsule] subject {sid} done. Outputs in {out}:", flush=True)
    for f in sorted(out.glob(f"{sid}_*.parquet")):
        print("    ", f.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
