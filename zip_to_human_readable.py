"""Create a human-readable CSV directly from the UAVCAN dataset ZIP file.

Each input entry must be named ``typeN_label.bin`` and contain records such as:

    Normal (000.192053)  can0  104E2001   [2]  00 C0

The generated CSV contains one row per CAN frame. Label 0 means Normal;
labels 1..N identify the attack type from the source filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
import random
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
FILE_PATTERN = re.compile(r"(?:^|/)type(?P<attack_type>\d+)_label\.bin$", re.IGNORECASE)
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
    timestamp: float
    interface: str
    can_id_decimal: int
    dlc: int
    data_decimal: list[int]
    label: int


def find_entries(archive: zipfile.ZipFile) -> list[tuple[int, str]]:
    entries = []
    for entry in archive.infolist():
        match = FILE_PATTERN.search(entry.filename.replace("\\", "/"))
        if match:
            entries.append((int(match.group("attack_type")), entry.filename))
    if not entries:
        raise FileNotFoundError("The ZIP contains no typeN_label.bin files.")
    return sorted(entries)


def parse_record(line: str, attack_type: int) -> dict[str, object] | None:
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


def iter_records(
    archive: zipfile.ZipFile, entries: list[tuple[int, str]]
):
    for attack_type, entry_name in entries:
        with archive.open(entry_name, "r") as compressed_file:
            for raw_line in compressed_file:
                line = raw_line.decode("utf-8", errors="replace")
                record = parse_record(line, attack_type)
                if record is not None:
                    yield record


def extract_records(
    archive: zipfile.ZipFile, entries: list[tuple[int, str]]
) -> tuple[list[Record], int]:
    records: list[Record] = []
    skipped = 0
    for attack_type, entry_name in entries:
        with archive.open(entry_name, "r") as compressed_file:
            for raw_line in compressed_file:
                line = raw_line.decode("utf-8", errors="replace")
                record = parse_record(line, attack_type)
                if record is None:
                    skipped += 1
                else:
                    records.append(record)
    return records, skipped


def clean_records(records: list[Record]) -> list[Record]:
    return [record for record in records if record.dlc == len(record.data_decimal)]


def balance_records(records: list[Record], seed: int) -> list[Record]:
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
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return ((features - mean) / scale).astype(np.float32), mean, scale


def write_csv(output_path: Path, records: list[Record]) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zip_file",
        type=Path,
        nargs="?",
        default=Path("UAVCAN_Attack_dataset(DroneCAN).zip"),
        help="Path to the UAVCAN dataset ZIP file (default: UAVCAN_Attack_dataset(DroneCAN).zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_data/uavcan_human_readable.csv"),
        help="Output CSV path (default: processed_data/uavcan_human_readable.csv).",
    )
    args = parser.parse_args()

    try:
        with zipfile.ZipFile(args.zip_file) as archive:
            entries = find_entries(archive)
            print("1. Data extraction: found input files in ZIP")
            extracted_records, skipped_count = extract_records(archive, entries)
    except KeyboardInterrupt:
        raise SystemExit("Conversion interrupted.")

    print(f"2. Parsing: extracted {len(extracted_records):,} valid records")
    records = clean_records(extracted_records)
    print(f"3. Cleaning: retained {len(records):,}; skipped {len(extracted_records) - len(records):,}")
    if not records:
        raise ValueError("No valid CAN records were found in the ZIP file.")

    print("4. Feature extraction: timestamp, interface, CAN ID, DLC, and decimal data")
    features = numeric_features(records)
    print("5. Label encoding: Normal=0; Attack labels retain their attack type")
    labels_before_balance = np.asarray([record.label for record in records], dtype=np.int64)
    print("6. Normalization / scaling: standardizing numeric model features")
    _, _, _ = normalize_features(features)
    print("7. Class balancing: undersampling the majority class")
    records = balance_records(records, seed=42)
    features = numeric_features(records)
    scaled_features, mean, scale = normalize_features(features)
    labels = np.asarray([record.label for record in records], dtype=np.int64)

    order = list(range(len(records)))
    random.Random(42).shuffle(order)
    features = features[order]
    scaled_features = scaled_features[order]
    labels = labels[order]
    records = [records[index] for index in order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, records)
    npz_path = args.output.with_name("uavcan_processed.npz")
    np.savez_compressed(
        npz_path,
        X=features,
        y=labels,
        feature_names=np.asarray(NUMERIC_FEATURE_NAMES),
        interfaces=np.asarray([record.interface for record in records]),
        mean=mean,
        scale=scale,
    )

    label_map = {"0": "Normal"}
    label_map.update({str(attack_type): f"Attack_type_{attack_type}" for attack_type, _ in entries})
    summary = {
        "source_zip": str(args.zip_file),
        "output_csv": str(args.output),
        "output_npz": str(npz_path),
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

    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Processed {len(records):,} balanced records; skipped {skipped_count:,} invalid lines.")
    print(f"Wrote {len(records):,} records to {args.output}")
    print(f"Saved: {npz_path}")
    print(f"Wrote metadata to {summary_path}")


if __name__ == "__main__":
    main()
