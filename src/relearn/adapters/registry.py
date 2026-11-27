from importlib import import_module

MODEL_NAMES = ("hvq_trans", "dinomaly_small", "u_cat", "rtdetrv2_s", "yolo11s", "ecf_yolov7_tiny")


def build_dataset(model, manifest, split, augment=False):
    if model not in MODEL_NAMES:
        raise ValueError(f"모델은 {MODEL_NAMES} 중 선택하세요")
    return import_module(f"relearn.adapters.{model}").Adapter(manifest, split, augment)


def build_loader(model, manifest, split, batch_size=1, workers=0, augment=False):
    import torch
    from torch.utils.data import DataLoader
    # Upstream imports stay isolated per environment; only the requested adapter loads.
    dataset = build_dataset(model, manifest, split, augment)
    if workers < 0 or batch_size < 1:
        raise ValueError("workers >= 0, batch_size >= 1 필요")
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=split == "train",
                      collate_fn=dataset.collate, persistent_workers=False,
                      generator=torch.Generator().manual_seed(int(manifest.config.get("seed", 42))))
