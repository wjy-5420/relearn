import torch
from .common import BaseAdapter


class Adapter(BaseAdapter):
    """HVQ-Trans custom_dataset keys, without its hardcoded MVTec class list."""
    def __getitem__(self, index):
        s = self.source[index]
        if self.split != "train" and not s["mask_valid"]:
            raise ValueError("HVQ-Trans 평가에는 유효 마스크가 필요합니다. 마스크 누락을 정상 GT로 대체하지 않습니다")
        return {"image": self.image(s), "mask": torch.from_numpy(s["mask"].copy()),
                "mask_valid": s["mask_valid"], "label": s["image_label"],
                "clsname": s["item"], "clslabel": self.item_id(s), "filename": s["image_path"],
                "height": s["geometry"]["original_height"], "width": s["geometry"]["original_width"]}
