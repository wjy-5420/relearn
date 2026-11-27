import torch
from .common import BaseAdapter


class Adapter(BaseAdapter):
    """Official RealIADDataset train/eval tuple contract; fixed 448 input, no 392 crop."""
    def __getitem__(self, index):
        s = self.source[index]
        image = self.image(s)
        if self.split == "train":
            return image, s["image_label"]
        if not s["mask_valid"]:
            raise ValueError("Dinomaly 공식 평가 tuple에는 유효 마스크가 필요합니다. 불량 마스크 누락을 먼저 검토하세요")
        return image, torch.from_numpy(s["mask"].copy()), s["image_label"], s["image_path"]
