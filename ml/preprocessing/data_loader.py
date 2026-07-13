import csv
import json
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json"}


def _read_with_fallback(file_path: Path, encoding: str) -> str:
    with file_path.open("r", encoding=encoding, newline="") as handle:
        return handle.read()


def iter_raw_datasets(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_dataset(file_path: Path) -> list[dict[str, Any]]:
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            for key in ("data", "records", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, dict)]

        return []

    delimiter = "," if suffix == ".csv" else "\t"

    csv.field_size_limit(10_000_000)

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with file_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                return [dict(row) for row in reader]
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode file")
