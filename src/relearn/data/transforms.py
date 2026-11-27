"""Shared geometric contract for all six models. No upstream augmentation runs here."""
from dataclasses import asdict, dataclass
import math
import random

import numpy as np
from PIL import Image, ImageEnhance

from .labels import valid_box


@dataclass(frozen=True)
class Geometry:
    original_width: int
    original_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    size: int
    angle_degrees: float = 0.0

    def inverse_boxes(self, boxes):
        if self.angle_degrees:
            raise ValueError("학습 회전 BBox의 역변환은 정확한 원본 박스를 복원하지 않습니다. 평가에서는 회전이 0입니다")
        out = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale_x + self.crop_x
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale_y + self.crop_y
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.original_width)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.original_height)
        return out

    def to_dict(self):
        return asdict(self)


def prepare(image, mask, boxes, record, config, epoch=0, augment=False):
    ow, oh = image.size
    crop = (0, 0, ow, oh)
    if config.get("crop_to_roi", False):
        if record.roi_xyxy is None or not record.roi_verified:
            raise ValueError("공통 ROI crop에는 검증된 ROI가 필요합니다")
        valid_box(record.roi_xyxy, ow, oh)
        # Integer crop boundaries are recorded for reversible evaluation coordinates.
        crop = (math.floor(record.roi_xyxy[0]), math.floor(record.roi_xyxy[1]),
                math.ceil(record.roi_xyxy[2]), math.ceil(record.roi_xyxy[3]))
        if len(boxes) and ((boxes[:, 0] < crop[0]).any() or (boxes[:, 1] < crop[1]).any()
                           or (boxes[:, 2] > crop[2]).any() or (boxes[:, 3] > crop[3]).any()):
            raise ValueError("공통 ROI가 라벨 BBox를 잘라냅니다")
        if mask is not None:
            original_pixels = int((np.asarray(mask) > 0).sum())
            if int((np.asarray(mask.crop(crop)) > 0).sum()) != original_pixels:
                raise ValueError("공통 ROI가 결함 마스크를 잘라냅니다")
    image = image.crop(crop)
    if mask is not None:
        mask = mask.crop(crop)
    width, height = image.size
    size = int(config.get("input_size", 448))
    if size != 448:
        raise ValueError("공통 기준 입력은 448입니다. 변경은 전체 실험 계약에서 별도로 승인해야 합니다")
    scale = min(size / width, size / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    px, py = (size - rw) // 2, (size - rh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (px, py))
    out_mask = Image.new("L", (size, size), 0)
    if mask is not None:
        out_mask.paste(mask.resize((rw, rh), Image.Resampling.NEAREST), (px, py))
        if np.asarray(mask).any() and not np.asarray(out_mask).any():
            raise ValueError("리사이즈 과정에서 결함 마스크가 소실되었습니다")
    result_boxes = boxes.copy()
    if len(boxes):
        result_boxes[:, [0, 2]] = (result_boxes[:, [0, 2]] - crop[0]) * (rw / width) + px
        result_boxes[:, [1, 3]] = (result_boxes[:, [1, 3]] - crop[1]) * (rh / height) + py
    angle = 0.0
    if augment:
        rng = random.Random(int(config.get("seed", 42)) + record.row * 1_000_003 + epoch * 10_000_019)
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(.9, 1.1))
        canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(.9, 1.1))
        angle = rng.uniform(-5, 5)
        before = np.asarray(out_mask).any()
        canvas = canvas.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(114, 114, 114))
        out_mask = out_mask.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
        if before and not np.asarray(out_mask).any():
            raise ValueError("회전 과정에서 결함이 소실되었습니다")
        if len(boxes):
            x1, y1, x2, y2 = result_boxes.T
            corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                                np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1) - size / 2
            theta = math.radians(angle)
            rotation = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            corners = corners @ rotation.T + size / 2
            if (corners < 0).any() or (corners > size).any():
                raise ValueError("회전이 객체 BBox를 자릅니다. 해당 ROI/증강 정책을 검토하세요")
            result_boxes = np.concatenate([corners.min(1), corners.max(1)], 1).astype(np.float32)
    geometry = Geometry(ow, oh, crop[0], crop[1], width, height, rw / width, rh / height, px, py, size, angle)
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return pixels, (np.asarray(out_mask) > 0).astype(np.float32)[None], result_boxes, geometry
