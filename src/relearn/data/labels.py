"""Verified class dictionary; no inferred IDs or invented normal-object boxes."""
import json
from pathlib import Path

import numpy as np


class ClassMap:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        entries = data["classes"]
        if data.get("status") != "verified" or not data.get("version"):
            raise ValueError("기존 검증 라벨에서 확정한 class_map(status=verified)이 필요합니다")
        if len(entries) != 141 or {x["id"] for x in entries} != set(range(141)):
            raise ValueError("class_map은 중복 없는 ID 0–140, 141개여야 합니다")
        self.classes = {entry["id"]: entry for entry in entries}
        self.item_ids = data["item_ids"]
        if len(self.item_ids) != 30 or set(self.item_ids.values()) != set(range(30)):
            raise ValueError("item_ids는 기존 품목 사전의 30개 품목 → 0–29 매핑이어야 합니다")
        if len({(entry["item"], entry["name"], entry["kind"]) for entry in entries}) != 141:
            raise ValueError("같은 품목·종류 내 표준 클래스 이름이 중복되었습니다")
        for entry in entries:
            expected = "defect" if entry["id"] < 111 else "normal"
            if entry["kind"] != expected or entry["item"] not in self.item_ids:
                raise ValueError(f"class_map 종류·품목 불일치: {entry}")
        self.normal_ids = {}
        for entry in entries:
            if entry["kind"] == "normal":
                if entry["item"] in self.normal_ids:
                    raise ValueError("품목별 정상 클래스는 1개여야 합니다")
                self.normal_ids[entry["item"]] = entry["id"]
        if set(self.normal_ids) != set(self.item_ids):
            raise ValueError("품목별 정상 클래스가 누락되었습니다")
        self.version = data["version"]

    def check(self, class_id, item, kind):
        entry = self.classes.get(class_id)
        if entry is None or entry["item"] != item or entry["kind"] != kind:
            raise ValueError(f"클래스 ID/품목/라벨 불일치: {class_id}, {item}, {kind}")


def valid_box(box, width, height):
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError(f"잘못된 BBox: {box}")
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"빈 박스 또는 원본 경계 초과: {box} in {width}×{height}")
    return [x1, y1, x2, y2]


def component_boxes(mask):
    """8-connected component boxes, half-open xyxy pixel coordinates."""
    from scipy.ndimage import find_objects, label
    components, _ = label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    return [[float(s[1].start), float(s[0].start), float(s[1].stop), float(s[0].stop)]
            for s in find_objects(components) if s is not None]


def annotations_for(record, width, height, mask, classes):
    if record.image_label not in (0, 1):
        raise ValueError(f"{record.row}행: UNKNOWN 라벨로 검출 학습·평가 불가")
    kind = "normal" if record.image_label == 0 else "defect"
    boxes, labels = [], []
    if record.annotations:
        for annotation in record.annotations:
            class_id = annotation["class_id"]
            if not isinstance(class_id, int):
                raise ValueError("class_id는 정수여야 합니다")
            classes.check(class_id, record.item, kind)
            boxes.append(valid_box(annotation["bbox"], width, height))
            labels.append(class_id)
    elif record.image_label == 0:
        if record.roi_xyxy is None or not record.roi_verified:
            raise ValueError(f"{record.row}행: 정상 객체의 검증된 ROI 또는 명시적 정상 annotations 필요")
        boxes.append(valid_box(record.roi_xyxy, width, height))
        labels.append(classes.normal_ids[record.item])
    else:
        if mask is None or record.defect_class_id is None:
            raise ValueError(f"{record.row}행: 결함 annotations 또는 마스크+검증 defect_class_id 필요")
        classes.check(record.defect_class_id, record.item, "defect")
        boxes = component_boxes(mask)
        if not boxes:
            raise ValueError(f"{record.row}행: 불량 이미지의 마스크가 비어 있습니다")
        labels = [record.defect_class_id] * len(boxes)
    unique = {(tuple(b), label) for b, label in zip(boxes, labels)}
    if len(unique) != len(boxes):
        raise ValueError(f"{record.row}행: 중복 BBox")
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)
