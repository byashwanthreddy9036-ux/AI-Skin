"""Dataset download registry (official ISIC Challenge S3 bucket).

Downloads stream to disk in chunks (never fully in memory), resume is not
attempted, and archives are extracted with a zip-slip guard. Anything larger
than :data:`CONFIRM_ABOVE_BYTES` requires explicit confirmation.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from skin_research.utils.logging import get_logger

log = get_logger(__name__)

ISIC_BASE = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
CONFIRM_ABOVE_BYTES = 100 * 2**20


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size_bytes: int
    extract: bool = True

    @property
    def url(self) -> str:
        return ISIC_BASE + self.name


@dataclass(frozen=True)
class DatasetSource:
    key: str
    description: str
    files: tuple[RemoteFile, ...]
    image_dir: str          # folder (inside raw/<key>) that holds the images
    groundtruth_csv: str    # path (inside raw/<key>) of the ground-truth CSV
    lesion_groupings_csv: str | None = None
    leakage_warning: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


_TRAIN_LEAK = (
    "The default model was TRAINED on these images (ISIC 2018 training set = HAM10000). "
    "Metrics on this data measure training-set fit, not generalisation."
)

DATASETS: dict[str, DatasetSource] = {
    "isic2018_val": DatasetSource(
        key="isic2018_val",
        description="ISIC 2018 Task 3 Validation: 193 dermoscopic images, held out from training",
        files=(
            RemoteFile("ISIC2018_Task3_Validation_Input.zip", 53_292_381),
            RemoteFile("ISIC2018_Task3_Validation_GroundTruth.zip", 7_619),
        ),
        image_dir="ISIC2018_Task3_Validation_Input",
        groundtruth_csv="ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
    ),
    "isic2018_test": DatasetSource(
        key="isic2018_test",
        description="ISIC 2018 Task 3 Test: 1,512 dermoscopic images, held out from training",
        files=(
            RemoteFile("ISIC2018_Task3_Test_Input.zip", 420_586_171),
            RemoteFile("ISIC2018_Task3_Test_GroundTruth.zip", 12_202),
        ),
        image_dir="ISIC2018_Task3_Test_Input",
        groundtruth_csv="ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
    ),
    "isic2018_train": DatasetSource(
        key="isic2018_train",
        description="ISIC 2018 Task 3 Training (= HAM10000): 10,015 images with lesion groupings",
        files=(
            RemoteFile("ISIC2018_Task3_Training_Input.zip", 2_771_732_744),
            RemoteFile("ISIC2018_Task3_Training_GroundTruth.zip", 36_681),
            RemoteFile("ISIC2018_Task3_Training_LesionGroupings.csv", 492_996, extract=False),
        ),
        image_dir="ISIC2018_Task3_Training_Input",
        groundtruth_csv="ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
        lesion_groupings_csv="ISIC2018_Task3_Training_LesionGroupings.csv",
        leakage_warning=_TRAIN_LEAK,
    ),
}


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, expected_bytes: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s (%s)", url, human_size(expected_bytes))
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        done, next_report = 0, 0.1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if expected_bytes and done / expected_bytes >= next_report:
                log.info("  %3.0f%%  %s", 100 * done / expected_bytes, human_size(done))
                next_report += 0.1
    if expected_bytes and tmp.stat().st_size != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {url}: expected {expected_bytes} bytes.")
    tmp.rename(dest)


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing members that would escape ``target`` (zip-slip)."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(target):
                raise RuntimeError(f"Unsafe path in archive {archive.name}: {member.filename}")
            if member.filename.startswith("__MACOSX/"):
                continue
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)


def download_dataset(key: str, raw_dir: Path, assume_yes: bool = False, keep_archives: bool = False) -> Path:
    """Download and extract a registered dataset into ``raw_dir/<key>``."""
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{key}'. Available: {', '.join(DATASETS)}")
    source = DATASETS[key]
    target = raw_dir / key
    target.mkdir(parents=True, exist_ok=True)
    pending = [f for f in source.files if not _is_present(f, target)]
    if not pending:
        log.info("Dataset '%s' already present in %s", key, target)
        return target
    total = sum(f.size_bytes for f in pending)
    print(f"{source.description}\nWill download {len(pending)} file(s), {human_size(total)} → {target}")
    if source.leakage_warning:
        print(f"WARNING: {source.leakage_warning}")
    if total > CONFIRM_ABOVE_BYTES and not assume_yes:
        answer = input(f"This is a large download ({human_size(total)}). Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("Download cancelled by user.")
    for remote in pending:
        dest = target / remote.name
        _download(remote.url, dest, remote.size_bytes)
        if remote.extract:
            log.info("Extracting %s", remote.name)
            safe_extract(dest, target)
            if not keep_archives:
                dest.unlink()
    return target


def _is_present(remote: RemoteFile, target: Path) -> bool:
    if not remote.extract:
        return (target / remote.name).is_file()
    folder = target / remote.name.removesuffix(".zip")
    return folder.is_dir() and any(folder.iterdir())
