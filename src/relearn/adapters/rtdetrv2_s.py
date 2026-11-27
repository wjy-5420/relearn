import torch
from .common import BaseAdapter, detection_collate, normalized_cxcywh


class Adapter(BaseAdapter):
    """Post-transform RT-DETRv2 target: normalized cxcywh, unremapped class IDs."""
    task = "detection"
    collate = staticmethod(detection_collate)

    def __getitem__(self, index):
        s = self.source[index]
        boxes = normalized_cxcywh(s["boxes"])
        target = {"boxes": boxes, "labels": torch.from_numpy(s["labels"].copy()),
                  "image_id": torch.tensor([s["row"]], dtype=torch.int64),
                  "area": torch.from_numpy((s["boxes"][:, 2:] - s["boxes"][:, :2]).prod(1).copy()),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                  "orig_size": torch.tensor([s["geometry"]["original_width"], s["geometry"]["original_height"]]),
                  "size": torch.tensor([448, 448])}
        # RT-DETRv2 upstream ToTensor scales to [0,1] and does not apply ImageNet normalization.
        return torch.from_numpy(s["image"].copy()), target
