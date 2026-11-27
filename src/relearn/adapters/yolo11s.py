import torch
from .common import BaseAdapter, normalized_cxcywh


class Adapter(BaseAdapter):
    """Ultralytics training batch fields; uint8 image avoids a second /255 error."""
    task = "detection"

    def __getitem__(self, index):
        s = self.source[index]
        g = s["geometry"]
        return {"img": torch.from_numpy((s["image"] * 255).round().astype("uint8")),
                "cls": torch.from_numpy(s["labels"].copy()).float().reshape(-1, 1),
                "bboxes": normalized_cxcywh(s["boxes"]), "batch_idx": torch.zeros(len(s["labels"])),
                "im_file": s["image_path"],
                "ori_shape": (s["geometry"]["original_height"], s["geometry"]["original_width"]),
                "resized_shape": (448, 448),
                "ratio_pad": ((g["scale_y"], g["scale_x"]), (g["pad_x"], g["pad_y"])),
                "geometry": g}

    @staticmethod
    def collate(batch):
        return {"img": torch.stack([s["img"] for s in batch]),
                "cls": torch.cat([s["cls"] for s in batch]),
                "bboxes": torch.cat([s["bboxes"] for s in batch]),
                "batch_idx": torch.cat([s["batch_idx"] + i for i, s in enumerate(batch)]),
                "im_file": [s["im_file"] for s in batch],
                "ori_shape": [s["ori_shape"] for s in batch],
                "resized_shape": [s["resized_shape"] for s in batch],
                "ratio_pad": [s["ratio_pad"] for s in batch],
                "geometry": [s["geometry"] for s in batch]}
