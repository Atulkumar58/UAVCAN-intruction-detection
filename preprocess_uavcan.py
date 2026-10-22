"""Convert the DroneCAN attack dataset into numeric deep-learning inputs.

Expected input files are named ``typeN_label.bin`` and contain candump-like
text records, for example:

    Normal (000.192053)  can0  104E2001   [2]  00 C0

The output label is 0 for Normal and N for an Attack record from typeN.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RECORD_PATTERN = re.compile(
    r"^(?P<kind>Normal|Attack)\s+"
    r"\((?P<timestamp>[-+]?\d+(?:\.\d+)?)\)\s+"
    r"(?P<interface>\S+)\s+"
    r"(?P<can_id>[0-9A-Fa-f]+)\s+"
    r"\[(?P<dlc>\d+)\]\s*"
    r"(?P<data>[0-9A-Fa-f\s]*)\s*$"
)
FILE_PATTERN = re.compile(r"type(?P<attack_type>\d+)_label\.bin$", re.IGNORECASE)

OUTPUT_FIELDS = ["timestamp", "interface", "can_id_decimal", "dlc", "data_decimal", "label"]
NUMERIC_FEATURE_NAMES = [
    "timestamp",
    "can_id_decimal",
    "dlc",
    "data_decimal_0",
    "data_decimal_1",
    "data_decimal_2",
    "data_decimal_3",
    "data_decimal_4",
    "data_decimal_5",
    "data_decimal_6",
    "data_decimal_7",
]


@dataclass(frozen=True)
class Record:
    """Represent one parsed UAVCAN frame and its attack label."""

    timestamp: float
    interface: str
    can_id_decimal: int
    dlc: int
    data_decimal: list[int]
    label: int


def find_input_files(input_dir: Path) -> list[tuple[int, Path]]:
    """Find typeN_label.bin files and return their attack types and paths."""

    files = []
    for path in sorted(input_dir.rglob("*.bin")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            files.append((int(match.group("attack_type")), path))
    if not files:
        raise FileNotFoundError(
            f"No typeN_label.bin files were found below {input_dir}"
        )
    return sorted(files)


def parse_record(line: str, attack_type: int) -> Record | None:
    """Parse one candump-style line into a Record, or return None if invalid."""

    match = RECORD_PATTERN.match(line.strip())
    if match is None:
        return None

    data_tokens = match.group("data").split()
    dlc = int(match.group("dlc"))
    if dlc > 8 or len(data_tokens) != dlc:
        return None

    data = [int(token, 16) for token in data_tokens]
    label = 0 if match.group("kind") == "Normal" else attack_type
    return Record(
        timestamp=float(match.group("timestamp")),
        interface=match.group("interface"),
        can_id_decimal=int(match.group("can_id"), 16),
        dlc=dlc,
        data_decimal=data,
        label=label,
    )


def extract_records(files: list[tuple[int, Path]]) -> tuple[list[Record], int]:
    """Read all input files and return parsed records plus invalid-line count."""

    records: list[Record] = []
    skipped = 0
    for attack_type, path in files:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                record = parse_record(line, attack_type)
                if record is None:
                    skipped += 1
                else:
                    records.append(record)
    return records, skipped


def clean_records(records: list[Record]) -> list[Record]:
    """Keep records whose declared DLC matches the number of payload bytes."""

    return [record for record in records if record.dlc == len(record.data_decimal)]


def balance_records(records: list[Record], seed: int) -> list[Record]:
    """Undersample the majority binary class so Normal and Attack are equal."""

    by_class: dict[int, list[Record]] = {}
    for record in records:
        by_class.setdefault(int(record.label > 0), []).append(record)
    if len(by_class) < 2:
        return records
    target = min(len(class_records) for class_records in by_class.values())
    rng = random.Random(seed)
    balanced = [record for class_records in by_class.values() for record in rng.sample(class_records, target)]
    rng.shuffle(balanced)
    return balanced


def numeric_features(records: list[Record]) -> np.ndarray:
    """Convert records to fixed-width numeric features, padding payloads to 8 bytes."""

    return np.asarray(
        [
            [
                record.timestamp,
                record.can_id_decimal,
                record.dlc,
                *record.data_decimal,
                *([0] * (8 - record.dlc)),
            ]
            for record in records
        ],
        dtype=np.float32,
    )


def normalize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize each numeric feature and return scaled data, means, and scales."""

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return ((features - mean) / scale).astype(np.float32), mean, scale


def write_csv(
    output_path: Path,
    records: list[Record],
) -> None:
    """Write the requested human-readable six-field representation to CSV."""

    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(OUTPUT_FIELDS)
        for record in records:
            writer.writerow(
                [
                    record.timestamp,
                    record.interface,
                    record.can_id_decimal,
                    record.dlc,
                    json.dumps(record.data_decimal),
                    record.label,
                ]
            )


def main() -> None:
    """Run extraction, parsing, cleaning, feature preparation, scaling, and balancing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("_dataset_probe"),
        help="Directory containing extracted typeN_label.bin files (default: _dataset_probe).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory for processed outputs (default: processed_data).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a human-readable CSV; this can be a large file.",
    )
    args = parser.parse_args()

    files = find_input_files(args.input_dir)
    print("1. Data extraction: found input files")
    extracted_records, skipped_count = extract_records(files)
    print(f"2. Parsing: extracted {len(extracted_records):,} valid records")
    records = clean_records(extracted_records)
    print(f"3. Cleaning: retained {len(records):,}; skipped {len(extracted_records) - len(records):,}")
    if not records:
        raise ValueError("No valid CAN records were found in the input files.")

    print("4. Feature extraction: timestamp, interface, CAN ID, DLC, and decimal data")
    features = numeric_features(records)
    print("5. Label encoding: Normal=0; Attack labels retain their attack type")
    labels_before_balance = np.asarray([record.label for record in records], dtype=np.int64)
    print("6. Normalization / scaling: standardizing numeric model features")
    scaled_features, mean, scale = normalize_features(features)
    print("7. Class balancing: undersampling the majority class")
    records = balance_records(records, seed=42)
    features = numeric_features(records)
    scaled_features, mean, scale = normalize_features(features)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    order = list(range(len(records)))
    random.Random(42).shuffle(order)
    scaled_features = scaled_features[order]
    features = features[order]
    labels = labels[order]
    records = [records[index] for index in order]

    np.savez_compressed(
        args.output_dir / "uavcan_processed.npz",
        X=features,
        y=labels,
        feature_names=np.asarray(NUMERIC_FEATURE_NAMES),
        interfaces=np.asarray([record.interface for record in records]),
        mean=mean,
        scale=scale,
    )

    if args.csv:
        write_csv(args.output_dir / "uavcan_processed.csv", records)

    label_map = {"0": "Normal"}
    label_map.update({str(attack_type): f"Attack_type_{attack_type}" for attack_type, _ in files})
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    summary = {
        "input_files": [str(path) for _, path in files],
        "records_extracted": int(len(extracted_records)),
        "records_before_balancing": int(len(labels_before_balance)),
        "records_written": int(len(records)),
        "records_skipped": int(skipped_count),
        "output_fields": OUTPUT_FIELDS,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "label_map": label_map,
        "class_counts": {str(label): int(np.sum(labels == label)) for label in np.unique(labels)},
        "random_seed": 42,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Processed {len(records):,} balanced records; skipped {skipped_count:,} invalid lines.")
    print(f"Saved: {args.output_dir / 'uavcan_processed.npz'}")


if __name__ == "__main__":
    main()