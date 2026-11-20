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


# --------------------------------------------------------------------------- parser
COMMANDS: dict[str, tuple[Handler, str]] = {
    "doctor": (cmd_doctor, "Check the environment, dependencies, data and model availability."),
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
    """Per-command options (extended as commands are added)."""


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
