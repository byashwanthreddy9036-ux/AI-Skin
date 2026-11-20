"""Command-line interface.

Run ``python -m skin_research.cli --help`` (or ``skin-research --help``) to list
commands. Every command accepts ``--config`` to point at an alternative YAML file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable

from skin_research import __version__
from skin_research.config import AppConfig, ConfigError, load_config
from skin_research.utils.logging import get_logger, setup_logging

log = get_logger("cli")

Handler = Callable[[argparse.Namespace, AppConfig], int]


# --------------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Check Python, packages, hardware, disk, data and model availability."""
    import os
    import platform

    ok = True

    def report(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'OK' if good else '!!'}] {label:<22} {value}")

    print(f"skin-research {__version__} — environment check\n")
    py_ok = sys.version_info[:2] == (3, 11)
    report("Python", f"{platform.python_version()} ({sys.executable})", py_ok)
    report("Platform", platform.platform())
    report("CPU cores", str(os.cpu_count()))
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            mem = {line.split(":")[0]: int(line.split()[1]) for line in fh}
        report("RAM available", f"{mem['MemAvailable'] / 2**20:.1f} GiB of {mem['MemTotal'] / 2**20:.1f} GiB")
    except (OSError, KeyError, ValueError):
        report("RAM", "unknown (non-Linux)")
    free = shutil.disk_usage(cfg.root).free / 2**30
    report("Disk free", f"{free:.0f} GiB", free > 5)

    try:
        import cv2
        import torch
        import torchvision

        report("torch", f"{torch.__version__} (threads={torch.get_num_threads()})")
        report("torchvision", torchvision.__version__)
        report("opencv", cv2.__version__)
        report("CUDA", "available" if torch.cuda.is_available() else "not available (CPU mode, expected)")
    except ImportError as exc:
        report("packages", f"missing: {exc}", False)

    report("Config", str(cfg.source_file))
    meta = cfg.dataset.metadata_csv
    report("Dataset metadata", f"{meta.relative_to(cfg.root) if meta.is_relative_to(cfg.root) else meta}"
           + ("" if meta.exists() else "  (not prepared yet: run download-data + prepare-data)"), True)

    try:
        from skin_research.inference.model_loader import cached_model_files

        files = cached_model_files(cfg)
        report("Model weights", f"{cfg.model.repo_id} " + ("cached" if files else "not downloaded yet (run download-model)"))
    except ImportError:
        pass

    print("\nAll checks passed." if ok else "\nSome checks need attention (marked !!).")
    return 0 if ok else 1


# --------------------------------------------------------------------------- data
def cmd_download_data(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Download a registered public dataset (asks before large downloads)."""
    from skin_research.data.download import DATASETS, download_dataset, human_size

    if args.list:
        for key, src in DATASETS.items():
            print(f"  {key:<16} {human_size(src.total_bytes):>9}  {src.description}")
        return 0
    key = args.dataset or cfg.dataset.name
    target = download_dataset(key, cfg.paths.raw_dir, assume_yes=args.yes)
    print(f"Dataset files are in {target}\nNext: skin-research prepare-data --dataset {key}")
    return 0


def cmd_prepare_data(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Convert raw dataset metadata into the canonical CSV used by every command."""
    import pandas as pd

    from skin_research.data import metadata as md
    from skin_research.data.download import DATASETS

    output = cfg.resolve(args.output) if args.output else cfg.dataset.metadata_csv
    if args.ham10000_metadata:
        df = md.convert_ham10000_metadata(cfg.resolve(args.ham10000_metadata))
        print("NOTE: the default model was trained on HAM10000; metrics on it are not held-out results.")
    else:
        key = args.dataset or cfg.dataset.name
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset '{key}'. Use --ham10000-metadata for HAM10000 or write your own CSV.")
        src = DATASETS[key]
        base = cfg.paths.raw_dir / key
        groupings = base / src.lesion_groupings_csv if src.lesion_groupings_csv else None
        df = md.convert_isic2018_groundtruth(base / src.groundtruth_csv, lesion_groupings_csv=groupings)
        if src.leakage_warning:
            print(f"WARNING: {src.leakage_warning}")
    if args.fetch_lesion_ids:
        ids = md.fetch_isic_lesion_ids(df["image_id"].tolist(), cfg.paths.processed_dir / "isic_api_cache.csv")
        found = ids.set_index("image_id")
        df["lesion_id"] = df["lesion_id"].fillna(df["image_id"].map(found["lesion_id"]))
        df["patient_id"] = df["patient_id"].fillna(df["image_id"].map(found["patient_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows → {output}")
    print(pd.Series(df["label"]).value_counts().to_string())
    print(f"lesion_id available for {int(df['lesion_id'].notna().sum())}/{len(df)} rows")
    return 0


def cmd_validate_data(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Validate metadata and images: missing files, corrupt images, labels, duplicates."""
    import json

    from skin_research.data.metadata import read_metadata_csv
    from skin_research.data.validation import validate_dataset

    df = read_metadata_csv(cfg.dataset.metadata_csv)
    report = validate_dataset(df, cfg.dataset.image_root, cfg.dataset.labels)
    out = cfg.paths.processed_dir / f"{cfg.dataset.name}_validation.csv"
    cols = ["image_id", "image_path", "label", "lesion_id", "exists", "readable", "label_valid",
            "width", "height", "format", "sha256", "error", "is_valid"]
    report.table[cols].to_csv(out, index=False)
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    print(f"Per-image validation table → {out}")
    return 0 if report.ok else 1


def cmd_dataset_info(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Print class distribution, identifier coverage and split information."""
    import json

    from skin_research.data.loader import dataset_statistics, prepare_dataset
    from skin_research.data.metadata import LABEL_DESCRIPTIONS

    prepared = prepare_dataset(cfg, check_near_duplicates=False)
    stats = dataset_statistics(prepared.full, cfg.dataset.labels)
    stats["split_level"] = prepared.split_level
    stats["split_counts"] = prepared.full["split"].value_counts().to_dict()
    stats["evaluation_images"] = len(prepared.table)
    print(f"Dataset: {cfg.dataset.name}  ({cfg.dataset.metadata_csv})\n")
    for label in cfg.dataset.labels:
        print(f"  {label:<6} {stats['class_counts'][label]:>6}  {100 * stats['class_fractions'][label]:5.1f}%  "
              f"{LABEL_DESCRIPTIONS.get(label, '')}")
    print()
    print(json.dumps({k: v for k, v in stats.items() if k not in {"class_counts", "class_fractions"}}, indent=2))
    out = cfg.paths.processed_dir / f"{cfg.dataset.name}_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    return 0


# --------------------------------------------------------------------------- quality
def cmd_assess_quality(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Compute quality metrics for dataset images (or given files) and write a report."""
    from pathlib import Path

    from skin_research.quality.assessment import assess_paths, flag_counts, summarize_quality

    if args.images:
        paths = []
        for item in args.images:
            p = Path(item)
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}) if p.is_dir() else [p]
        ids, labels, name = [p.stem for p in paths], None, "custom"
    else:
        from skin_research.data.loader import prepare_dataset

        prepared = prepare_dataset(cfg, check_near_duplicates=False)
        paths, ids, name = prepared.paths, prepared.table["image_id"].tolist(), cfg.dataset.name
        labels = prepared.table["label"].tolist()
    if not paths:
        raise ValueError("No images to assess.")
    df = assess_paths(paths, cfg.quality, ids)
    if labels is not None:
        df.insert(1, "label", labels)
    out_dir = cfg.paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"quality_{name}.csv"
    df.to_csv(csv, index=False)
    summary = summarize_quality(df[df["valid"]])
    flags = flag_counts(df)
    lines = [f"# Image quality report — {name}", "",
             f"Images assessed: {len(df)} (valid: {int(df['valid'].sum())})", "",
             "Metrics are classical no-reference heuristics. Blur/sharpness scores are computed after resizing "
             f"to {cfg.quality.get('analysis_long_side', 512)} px on the long side and are only comparable "
             "between images of similar content; no single score is a universal quality verdict.", "",
             "## Metric distribution", "", summary.round(4).to_markdown(), "",
             "## Heuristic quality flags", ""]
    lines += [f"- {k}: {v}" for k, v in flags.items()] or ["- none raised"]
    if labels is not None:
        lines += ["", "## Mean metrics by class", "",
                  summarize_quality(df[df["valid"]], by="label").round(4).to_markdown()]
    md = out_dir / f"quality_{name}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.round(4).to_string())
    print(f"\nFlags: {flags or 'none'}\nPer-image table → {csv}\nReport → {md}")
    return 0


# --------------------------------------------------------------------------- variations
def _apply_overrides(args: argparse.Namespace, cfg: AppConfig) -> None:
    """Command-line overrides shared by the pipeline commands."""
    if getattr(args, "max_images", None):
        cfg.dataset.max_images = args.max_images
        cfg.raw["dataset"]["max_images"] = args.max_images
    if getattr(args, "batch_size", None):
        cfg.model.batch_size = args.batch_size
        cfg.raw["model"]["batch_size"] = args.batch_size


def cmd_generate_variations(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Generate degraded and enhanced images with a traceability manifest."""
    from skin_research.data.loader import prepare_dataset
    from skin_research.preprocessing.variations import (
        build_conditions, generate_variations, manifest_path, select_conditions)

    _apply_overrides(args, cfg)
    prepared = prepare_dataset(cfg, check_near_duplicates=False)
    conditions = select_conditions(build_conditions(cfg), args.conditions)
    print(f"Generating {len(conditions)} condition(s) for {len(prepared.table)} image(s)")
    manifest = generate_variations(cfg, prepared.table, conditions, force=args.force)
    out = manifest_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = manifest.groupby("condition", sort=False).agg(
        images=("image_id", "size"), brightness=("brightness", "mean"), contrast=("rms_contrast", "mean"),
        laplacian_var=("laplacian_var", "mean"),
        questionable=("label_usefulness", lambda s: int((s == "questionable").sum())))
    print(summary.round(3).to_string())
    print(f"\nManifest ({len(manifest)} rows) → {out}")
    return 0


# --------------------------------------------------------------------------- model
def _load_predictor(cfg: AppConfig):
    from skin_research.inference.model_loader import check_label_compatibility, load_model
    from skin_research.inference.predictor import Predictor

    adapter = load_model(cfg)
    compat = check_label_compatibility(adapter.labels, cfg.model.label_map, cfg.dataset.labels)
    if not compat.ok:
        log.warning("Label mapping incompatible — predictions are saved but metrics will be refused: %s",
                    "; ".join(compat.problems))
    return adapter, Predictor(adapter, cfg.model.label_map, cfg.dataset.labels, cfg.model.batch_size), compat


def cmd_download_model(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Download and verify the configured pretrained model (~16 MB)."""
    import json

    from skin_research.inference.model_loader import check_label_compatibility, load_model

    adapter = load_model(cfg)
    meta = adapter.metadata()
    compat = check_label_compatibility(adapter.labels, cfg.model.label_map, cfg.dataset.labels)
    print(json.dumps({k: meta[k] for k in ("repo_id", "revision", "architecture", "input_size", "parameters",
                                           "labels", "weights_sha256", "license")}, indent=2))
    print(f"Label mapping to dataset: {'OK' if compat.ok else compat.problems}")
    return 0 if compat.ok else 1


def cmd_predict(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Run the model on given images, or on every image in the variations manifest."""
    from pathlib import Path

    import pandas as pd

    from skin_research.preprocessing.variations import manifest_path

    _apply_overrides(args, cfg)
    _, predictor, _ = _load_predictor(cfg)
    if args.images:
        paths = [Path(p) for p in args.images]
        df = predictor.predict_paths(paths)
        cols = ["path", "predicted_label", "confidence", "margin"]
        print(df[cols + [f"prob_{l}" for l in cfg.dataset.labels]].round(4).to_string(index=False))
        out = cfg.paths.predictions_dir / "adhoc_predictions.csv"
    else:
        manifest_file = manifest_path(cfg)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"No manifest at {manifest_file}. Run generate-variations first, "
                                    "or pass image paths to predict.")
        manifest = pd.read_csv(manifest_file)
        if args.conditions:
            manifest = manifest[manifest["condition"].apply(lambda c: c == "original" or any(
                c == n or c.startswith(n) for n in args.conditions))]
        manifest = manifest[manifest["output_path"].notna()]
        print(f"Predicting {len(manifest)} images from {manifest_file}")
        df = predictor.predict_paths([Path(p) for p in manifest["output_path"]], manifest.to_dict("records"))
        out = cfg.paths.predictions_dir / cfg.experiment.name / "predictions.csv"
        if "correct" in df:
            print(df.groupby("condition", sort=False)["correct"].mean().round(3).to_string())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} predictions ({predictor.throughput:.1f} img/s) → {out}")
    return 0


def cmd_demo(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Milestone demo: one real image → original, blurred, poorly lit and enhanced versions → predictions."""
    import shutil
    from pathlib import Path

    import pandas as pd

    from skin_research.data.loader import load_image, save_image
    from skin_research.data.metadata import read_metadata_csv
    from skin_research.preprocessing.degradation import apply_degradation
    from skin_research.preprocessing.enhancement import apply_enhancement
    from skin_research.quality.assessment import assess_array

    if args.image:
        source, label, image_id = Path(args.image), args.label, Path(args.image).stem
    else:
        meta = read_metadata_csv(cfg.dataset.metadata_csv)
        row = meta[meta["image_id"] == args.image_id].iloc[0] if args.image_id else meta.iloc[0]
        source, label, image_id = cfg.dataset.image_root / row["image_path"], row["label"], row["image_id"]
    sample = cfg.paths.sample_dir / source.name
    sample.parent.mkdir(parents=True, exist_ok=True)
    if not sample.exists():
        shutil.copy2(source, sample)  # read-only copy for smoke tests; the original stays untouched

    original = load_image(sample)
    variants = {
        "original": original,
        "gaussian_blur_moderate": apply_degradation(original, "gaussian_blur", cfg.degradations["gaussian_blur"]["moderate"]),
        "poor_lighting_moderate": apply_degradation(original, "underexposure", cfg.degradations["underexposure"]["moderate"]),
        "sharpened": apply_enhancement(original, "sharpen", cfg.enhancements["sharpen"]),
        "clahe": apply_enhancement(original, "clahe", cfg.enhancements["clahe"]),
    }
    out_dir = cfg.paths.outputs_dir / "demo" / image_id
    if out_dir.exists():
        shutil.rmtree(out_dir)  # generated demo output only
    records = []
    for name, img in variants.items():
        path = sample if name == "original" else save_image(img, out_dir / name)
        q = assess_array(img, cfg.quality)
        records.append({"image_id": image_id, "label": label, "version": name, "path": str(path),
                        "brightness": q["brightness"], "rms_contrast": q["rms_contrast"],
                        "laplacian_var": q["laplacian_var"], "quality_flags": q["quality_flags"]})
    adapter, predictor, _ = _load_predictor(cfg)
    frames = list(predictor.predict_stream(zip(records, variants.values())))
    df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "demo_predictions.csv"
    df.to_csv(csv, index=False)

    print(f"Image {image_id}  ground truth: {label}  model: {adapter.cfg.repo_id}\n")
    print(df[["version", "brightness", "rms_contrast", "laplacian_var", "predicted_label", "confidence"]]
          .round(3).to_string(index=False))
    print("\nClass probabilities:")
    print(df.set_index("version")[[f"prob_{l}" for l in cfg.dataset.labels]].round(3)
          .rename(columns=lambda c: c[5:]).to_string())
    base = df.iloc[0]
    print("\nComparison with original:")
    for _, r in df.iloc[1:].iterrows():
        change = "SAME class" if r.predicted_label == base.predicted_label else f"CHANGED → {r.predicted_label}"
        print(f"  {r.version:<24} {change:<18} p(true={label}) {base[f'prob_{label}']:.3f} → {r[f'prob_{label}']:.3f}" if label
              else f"  {r.version:<24} {change}")
    print(f"\nSaved → {csv}")
    return 0


# --------------------------------------------------------------------------- parser
COMMANDS: dict[str, tuple[Handler, str]] = {
    "doctor": (cmd_doctor, "Check the environment, dependencies, data and model availability."),
    "download-data": (cmd_download_data, "Download a public dataset from the official ISIC archive."),
    "prepare-data": (cmd_prepare_data, "Convert raw dataset metadata into the canonical CSV."),
    "validate-data": (cmd_validate_data, "Validate metadata and images (missing, corrupt, labels, duplicates)."),
    "dataset-info": (cmd_dataset_info, "Show dataset statistics, class balance and split information."),
    "assess-quality": (cmd_assess_quality, "Compute image quality metrics and write a quality report."),
    "generate-variations": (cmd_generate_variations, "Generate degraded/enhanced images with a traceability manifest."),
    "download-model": (cmd_download_model, "Download and verify the pretrained model weights (~16 MB)."),
    "predict": (cmd_predict, "Run the pretrained model on images or on the variations manifest."),
    "demo": (cmd_demo, "One-image milestone: original/blurred/dark/enhanced versions and their predictions."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skin-research",
        description="Skin lesion image quality vs. pretrained classifier performance — research CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config (default: configs/config.yaml).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, help_text) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_text, description=help_text)
        _add_command_args(name, sp)
    return parser


def _add_command_args(name: str, sp: argparse.ArgumentParser) -> None:
    """Per-command options."""
    if name == "download-data":
        sp.add_argument("--dataset", help="Dataset key (default: dataset.name from config).")
        sp.add_argument("--list", action="store_true", help="List available datasets and sizes.")
        sp.add_argument("-y", "--yes", action="store_true", help="Do not ask before large downloads.")
    elif name == "prepare-data":
        sp.add_argument("--dataset", help="Registered dataset key (default: dataset.name).")
        sp.add_argument("--ham10000-metadata", help="Path to HAM10000_metadata.csv (manual download).")
        sp.add_argument("--output", help="Output CSV (default: dataset.metadata_csv).")
        sp.add_argument("--fetch-lesion-ids", action="store_true",
                        help="Look up lesion_id/patient_id per image via the public ISIC Archive API.")
    if name in {"generate-variations", "predict", "experiment"}:
        sp.add_argument("--max-images", type=int, help="Approximate image limit: stratified and lesion-aware, keeps ≥1 lesion per class, so it can exceed N.")
        sp.add_argument("--conditions", nargs="+", help="Only these conditions (name or prefix); original is always kept.")
        sp.add_argument("--force", action="store_true", help="Regenerate variation folders whose parameters changed.")
    if name in {"predict", "experiment"}:
        sp.add_argument("--batch-size", type=int, help="Override model.batch_size.")
    if name == "predict":
        sp.add_argument("images", nargs="*", help="Image files (default: all images in the manifest).")
    if name == "demo":
        sp.add_argument("--image", help="Any image file (default: first image of the configured dataset).")
        sp.add_argument("--label", help="Ground-truth label for --image, if known.")
        sp.add_argument("--image-id", help="Pick a dataset image by id, e.g. ISIC_0034329.")
    if name == "assess-quality":
        sp.add_argument("images", nargs="*", help="Image files or folders (default: the configured dataset).")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level, cfg.paths.logs_dir / "skin_research.log")
    handler, _ = COMMANDS[args.command]
    try:
        return handler(args, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        log.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
