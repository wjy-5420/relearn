import torch
from .common import BaseAdapter, normalized_cxcywh


class Adapter(BaseAdapter):
    """YOLOv7 train.py tuple: uint8 image, [batch,class,cx,cy,w,h], path, shapes."""
    task = "detection"

    def __getitem__(self, index):
        s = self.source[index]
        labels = torch.zeros((len(s["labels"]), 6), dtype=torch.float32)
        labels[:, 1] = torch.from_numpy(s["labels"].copy())
        labels[:, 2:] = normalized_cxcywh(s["boxes"])
        g = s["geometry"]
        # Kept as neutral metadata: use Geometry.inverse_boxes after ROI/letterbox evaluation.
        shapes = ((g["original_height"], g["original_width"]),
                  ((g["scale_y"], g["scale_x"]), (g["pad_x"], g["pad_y"])))
        return torch.from_numpy((s["image"] * 255).round().astype("uint8")), labels, s["image_path"], shapes

    @staticmethod
    def collate(batch):
        images, labels, paths, shapes = zip(*batch)
        result = []
        for i, value in enumerate(labels):
            value = value.clone()
            value[:, 0] = i
            result.append(value)
        return torch.stack(images), torch.cat(result), paths, shapes
