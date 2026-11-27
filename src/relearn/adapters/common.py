import torch
from torch.utils.data import Dataset, default_collate

from relearn.data.dataset import CSVImages


class BaseAdapter(Dataset):
    task = "anomaly"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    collate = staticmethod(default_collate)

    def __init__(self, manifest, split, augment=False):
        self.source = CSVImages(manifest, split, self.task, augment)
        self.split = split

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def image(self, sample):
        image = torch.from_numpy(sample["image"].copy())
        return (image - image.new_tensor(self.mean)[:, None, None]) / image.new_tensor(self.std)[:, None, None]

    def item_id(self, sample):
        if self.source.classes is None:
            raise ValueError("품목 ID가 필요한 어댑터에는 기존 사전에서 검증한 class_map.item_ids가 필요합니다")
        return self.source.classes.item_ids[sample["item"]]


def normalized_cxcywh(boxes, size=448):
    boxes = torch.from_numpy(boxes.copy())
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), dim=1) / size


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
