"""In-memory CSV reader. This module does not generate a benchmark dataset."""
import numpy as np
from PIL import Image

from .labels import ClassMap, annotations_for
from .transforms import prepare


class CSVImages:
    def __init__(self, manifest, split, task="anomaly", augment=False):
        if split not in ("train", "val", "test") or task not in ("anomaly", "detection"):
            raise ValueError("split/task 값 오류")
        report = manifest.validate()
        if not report["valid"]:
            raise ValueError("CSV 계약 검사 실패: " + "; ".join(report["errors"][:10]))
        if augment and split != "train":
            raise ValueError("무작위 증강은 Train에서만 허용합니다")
        self.manifest, self.split, self.task, self.augment = manifest, split, task, augment
        self.records = manifest.select(split, normal_products_only=task == "anomaly" and split == "train")
        if not self.records:
            raise ValueError(f"{split}/{task}에 사용 가능한 데이터가 없습니다")
        path = manifest.config.get("class_map")
        self.classes = ClassMap(path) if path else None
        if task == "detection" and self.classes is None:
            raise ValueError("검출 어댑터에는 팀이 검증한 141클래스 사전이 필요합니다")
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if -1 in (record.product_label, record.image_label):
            raise ValueError(f"{record.row}행: UNKNOWN 샘플은 검토 완료 후 사용하세요")
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        mask_image = None
        if record.mask_path is not None:
            with Image.open(record.mask_path) as source:
                mask_image = source.convert("L")
            if mask_image.size != image.size:
                raise ValueError(f"{record.row}행: 원본 이미지/마스크 크기 불일치")
            if bool(np.asarray(mask_image).any()) != bool(record.image_label):
                raise ValueError(f"{record.row}행: 이미지 정상/불량 라벨과 마스크 불일치")
        boxes, labels = np.empty((0, 4), np.float32), np.empty((0,), np.int64)
        if self.task == "detection":
            boxes, labels = annotations_for(record, *image.size,
                                           np.asarray(mask_image) if mask_image is not None else None, self.classes)
        pixels, mask, boxes, geometry = prepare(image, mask_image, boxes, record, self.manifest.config,
                                               self.epoch, self.augment)
        return {"image": pixels, "mask": mask, "mask_valid": mask_image is not None or record.image_label == 0,
                "boxes": boxes, "labels": labels, "image_label": record.image_label,
                "product_label": record.product_label, "item": record.item, "product_id": record.product_id,
                "view": record.view, "image_path": str(record.image_path), "row": record.row,
                "geometry": geometry.to_dict()}
