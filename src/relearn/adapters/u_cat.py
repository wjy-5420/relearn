import torch
from .common import BaseAdapter


class Adapter(BaseAdapter):
    """Provisional neutral input contract. Not asserted to match unpublished U-CAT code."""
    def __getitem__(self, index):
        s = self.source[index]
        if self.split != "train" and not s["mask_valid"]:
            raise ValueError("U-CAT 평가 입력에는 유효 마스크가 필요합니다")
        return {"image": self.image(s), "mask": torch.from_numpy(s["mask"].copy()),
                "mask_valid": s["mask_valid"], "label": s["image_label"],
                "item_id": self.item_id(s), "path": s["image_path"]}
