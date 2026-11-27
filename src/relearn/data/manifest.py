"""Read the team's CSV without selecting, resplitting or writing its data."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("csv_path", "class_map"):
        if config.get(key):
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else path.parent / value)
    root = Path(config["data_root"])
    config["data_root"] = str(root if root.is_absolute() else path.parent / root)
    return config


@dataclass(frozen=True)
class Record:
    row: int
    image_path: Path
    item: str
    product_id: str
    view: str
    split: str
    product_label: int
    image_label: int
    mask_path: Path | None = None
    defect_class_id: int | None = None
    roi_xyxy: tuple | None = None
    roi_verified: bool = False
    annotations: tuple = ()

    @property
    def product_key(self):
        return self.item, self.product_id


def resolve_data_path(value, config):
    value = value.strip().replace("\\", "/")
    for source, destination in sorted(config.get("path_prefixes", {}).items(), key=lambda x: -len(x[0])):
        source = source.replace("\\", "/").rstrip("/")
        if value == source or value.startswith(source + "/"):
            value = destination.rstrip("/\\") + value[len(source):]
            break
    path = Path(value)
    return (path if path.is_absolute() else Path(config["data_root"]) / path).resolve()


class Manifest:
    def __init__(self, records, config, digest=""):
        self.records = tuple(records)
        self.config = config
        self.digest = digest

    @classmethod
    def read(cls, config):
        csv_path = Path(config["csv_path"])
        columns = config.get("columns", {})
        required = ("image_path", "item", "product_id", "view", "split", "product_label", "image_label")
        records = []
        with csv_path.open(encoding=config.get("encoding", "utf-8-sig"), newline="") as stream:
            reader = csv.DictReader(stream, delimiter=config.get("delimiter", ","))
            header = reader.fieldnames or []
            missing = [k for k in required if columns.get(k, k) not in header]
            if missing:
                raise ValueError(f"CSV 컬럼 매핑 필요: {missing}; 실제 헤더={header}")
            for row_num, row in enumerate(reader, 2):
                def get(key):
                    return (row.get(columns.get(key, key)) or "").strip()
                try:
                    for key in ("image_path", "item", "product_id", "view", "split"):
                        if not get(key):
                            raise ValueError(f"필수 값 누락: {key}")
                    labels = config.get("label_values", {"0": 0, "1": 1, "UNKNOWN": -1, "": -1})
                    product_label, image_label = int(labels[get("product_label")]), int(labels[get("image_label")])
                    if product_label not in (-1, 0, 1) or image_label not in (-1, 0, 1):
                        raise ValueError("라벨은 0/1/UNKNOWN이어야 합니다")
                    split = config.get("split_values", {"train": "train", "val": "val", "test": "test"})[get("split")]
                    if split not in ("train", "val", "test"):
                        raise ValueError("분할은 train/val/test이어야 합니다")
                    view = get("view").upper()
                    if view.isdigit():
                        view = "C" + view
                    if view not in ("C1", "C2", "C3", "C4", "C5"):
                        raise ValueError(f"유효하지 않은 시점: {view}")
                    roi = tuple(json.loads(get("roi_xyxy"))) if get("roi_xyxy") else None
                    if roi is not None and (len(roi) != 4 or not all(isinstance(x, (float, int)) for x in roi)):
                        raise ValueError("roi_xyxy는 원본 픽셀 좌표 [x1,y1,x2,y2]여야 합니다")
                    annotations = json.loads(get("annotations")) if get("annotations") else []
                    if not isinstance(annotations, list):
                        raise ValueError("annotations는 JSON 배열이어야 합니다")
                    records.append(Record(row_num, resolve_data_path(get("image_path"), config), get("item"),
                        get("product_id"), view, split, product_label, image_label,
                        resolve_data_path(get("mask_path"), config) if get("mask_path") else None,
                        int(get("defect_class_id")) if get("defect_class_id") else None,
                        roi, get("roi_verified").lower() in ("true", "1", "yes"), tuple(annotations)))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"CSV {row_num}행: {exc}") from exc
        if not records:
            raise ValueError("CSV에 데이터 행이 없습니다")
        return cls(records, config, hashlib.sha256(csv_path.read_bytes()).hexdigest())

    def validate(self, check_files=False):
        errors, warnings = [], []
        groups = defaultdict(list)
        paths = set()
        for record in self.records:
            groups[record.product_key].append(record)
            path_key = os.path.normcase(str(record.image_path))
            if path_key in paths:
                errors.append(f"중복 이미지 경로: {record.image_path}")
            paths.add(path_key)
            if record.product_label == 0 and record.image_label == 1:
                errors.append(f"{record.row}행: 정상 제품에 불량 시점 라벨")
            if record.image_label == -1 or record.product_label == -1:
                warnings.append(f"{record.row}행: UNKNOWN 라벨, 학습/평가 전에 검토 필요")
            if record.image_label == 1 and not record.mask_path:
                warnings.append(f"{record.row}행: 불량 시점의 마스크 없음 (Pixel 평가 불가)")
            if check_files:
                for kind, path in (("image", record.image_path), ("mask", record.mask_path)):
                    if path is not None and not path.is_file():
                        errors.append(f"{record.row}행: {kind} 파일 없음: {path}")
        for key, rows in groups.items():
            if len({r.split for r in rows}) != 1:
                errors.append(f"제품 분할 누출: {key}")
            if len({r.product_label for r in rows}) != 1:
                errors.append(f"제품 라벨 충돌: {key}")
            views = [r.view for r in rows]
            if len(views) != len(set(views)):
                errors.append(f"중복 시점: {key}")
            if self.config.get("require_five_views", True) and set(views) != {f"C{i}" for i in range(1, 6)}:
                errors.append(f"5시점 누락: {key}: {views}")
            if rows[0].product_label == 1 and all(r.image_label == 0 for r in rows):
                warnings.append(f"불량 제품의 모든 이미지가 정상 라벨: {key} (팀원 검토 필요)")
        return {"csv_sha256": self.digest, "images": len(self.records), "products": len(groups),
                "items": len({r.item for r in self.records}),
                "split_counts": dict(Counter(r.split for r in self.records)),
                "product_split_counts": dict(Counter(rows[0].split for rows in groups.values())),
                "image_label_counts": dict(Counter(str(r.image_label) for r in self.records)),
                "product_label_counts": dict(Counter(str(rows[0].product_label) for rows in groups.values())),
                "view_counts": dict(Counter(r.view for r in self.records)),
                "item_counts": dict(Counter(r.item for r in self.records)),
                "errors": errors, "warnings": warnings, "valid": not errors}

    def select(self, split, normal_products_only=False):
        return tuple(r for r in self.records if r.split == split and
                     (not normal_products_only or r.product_label == 0))
