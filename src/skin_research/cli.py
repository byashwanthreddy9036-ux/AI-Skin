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


# --------------------------------------------------------------------------- parser
COMMANDS: dict[str, tuple[Handler, str]] = {
    "doctor": (cmd_doctor, "Check the environment, dependencies, data and model availability."),
    "download-data": (cmd_download_data, "Download a public dataset from the official ISIC archive."),
    "prepare-data": (cmd_prepare_data, "Convert raw dataset metadata into the canonical CSV."),
    "validate-data": (cmd_validate_data, "Validate metadata and images (missing, corrupt, labels, duplicates)."),
    "dataset-info": (cmd_dataset_info, "Show dataset statistics, class balance and split information."),
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
