"""Synthetic, temporary CSVs only. Never extract/resplit the team's Real-IAD data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from relearn.data import Manifest
from relearn.data.dataset import CSVImages
from relearn.data.labels import ClassMap, component_boxes
from relearn.data.transforms import Geometry


class CSVFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        entries = [{"id": i, "name": f"test_defect_{i}", "item": f"item{i % 30}", "kind": "defect"}
                   for i in range(111)]
        entries += [{"id": 111 + i, "name": f"test_normal_{i}", "item": f"item{i}", "kind": "normal"}
                    for i in range(30)]
        dictionary = {"status": "verified", "version": "synthetic-test-only-v1", "classes": entries,
                      "item_ids": {f"item{i}": i for i in range(30)}}
        self.map_path = self.root / "test_classes.json"
        self.map_path.write_text(json.dumps(dictionary), encoding="utf-8")
        self.rows = []
        for product, split, label in (("good_train", "train", 0), ("bad_train", "train", 1),
                                      ("bad_val", "val", 1), ("good_test", "test", 0)):
            for view in range(1, 6):
                image_label = int(label == 1 and view == 3)
                stem = f"{product}_C{view}"
                Image.new("RGB", (128, 96), (100, 150, 200)).save(self.root / (stem + ".jpg"))
                mask = np.zeros((96, 128), dtype=np.uint8)
                if image_label:
                    mask[30:40, 40:50] = 255
                    mask[50:60, 70:80] = 255
                Image.fromarray(mask).save(self.root / (stem + ".png"))
                self.rows.append({"image_path": stem + ".jpg", "item": "item0", "product_id": product,
                                  "view": f"C{view}", "split": split, "product_label": str(label),
                                  "image_label": str(image_label), "mask_path": stem + ".png",
                                  "defect_class_id": "0" if image_label else "",
                                  "roi_xyxy": "[10,10,118,86]", "roi_verified": "true", "annotations": ""})
        self.csv_path = self.root / "shared.csv"
        self.config = {"csv_path": str(self.csv_path), "data_root": str(self.root),
                       "class_map": str(self.map_path), "input_size": 448, "seed": 42}
        self.write()

    def write(self, mapping=None):
        mapping = mapping or {}
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[mapping.get(k, k) for k in self.rows[0]])
            writer.writeheader()
            writer.writerows({mapping.get(k, k): v for k, v in row.items()} for row in self.rows)

    def manifest(self):
        return Manifest.read(self.config)


class ManifestTests(CSVFixture):
    def test_header_mapping_and_read_only(self):
        mapping = {"image_path": "이미지경로", "split": "분할", "product_id": "제품번호"}
        self.write(mapping)
        self.config["columns"] = mapping
        original = self.csv_path.read_bytes()
        manifest = self.manifest()
        self.assertTrue(manifest.validate(check_files=True)["valid"])
        self.assertEqual(len(manifest.records), 20)
        self.assertEqual(self.csv_path.read_bytes(), original)

    def test_product_split_leak_rejected(self):
        self.rows[0]["split"] = "test"
        self.write()
        self.assertTrue(any("분할 누출" in e for e in self.manifest().validate()["errors"]))

    def test_missing_view_rejected(self):
        self.rows.pop()
        self.write()
        self.assertFalse(self.manifest().validate()["valid"])

    def test_duplicate_image_rejected(self):
        self.rows[1]["image_path"] = self.rows[0]["image_path"]
        self.write()
        self.assertTrue(any("중복 이미지" in e for e in self.manifest().validate()["errors"]))

    def test_ad_filters_by_product_not_only_image_label(self):
        manifest = self.manifest()
        ad = CSVImages(manifest, "train", "anomaly")
        od = CSVImages(manifest, "train", "detection")
        self.assertEqual(len(ad), 5)
        self.assertEqual(len(od), 10)
        self.assertTrue(all(r.product_id == "good_train" for r in ad.records))

    def test_ad_does_not_silently_drop_unknown_views(self):
        self.rows[0]["image_label"] = "UNKNOWN"
        self.write()
        dataset = CSVImages(self.manifest(), "train", "anomaly")
        self.assertEqual(len(dataset), 5)
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            dataset[0]


class GeometryLabelTests(CSVFixture):
    def test_box_roundtrip_and_mask_alignment(self):
        dataset = CSVImages(self.manifest(), "val", "detection")
        s = dataset[2]
        restored = Geometry(**s["geometry"]).inverse_boxes(s["boxes"])
        np.testing.assert_allclose(restored, [[40, 30, 50, 40], [70, 50, 80, 60]], atol=1e-5)
        self.assertEqual(s["image"].shape, (3, 448, 448))
        self.assertEqual(s["mask"].shape, (1, 448, 448))
        self.assertEqual(s["labels"].tolist(), [0, 0])

    def test_normal_view_of_defect_product_has_normal_roi_label(self):
        s = CSVImages(self.manifest(), "val", "detection")[0]
        self.assertEqual(s["product_label"], 1)
        self.assertEqual(s["image_label"], 0)
        self.assertEqual(s["labels"].tolist(), [111])

    def test_normal_roi_never_invented(self):
        self.rows[0]["roi_verified"] = "false"
        self.write()
        with self.assertRaisesRegex(ValueError, "검증된 ROI"):
            CSVImages(self.manifest(), "train", "detection")[0]

    def test_missing_defect_mask_does_not_become_negative_gt(self):
        self.rows[12]["mask_path"] = ""
        self.write()
        s = CSVImages(self.manifest(), "val", "anomaly")[2]
        self.assertFalse(s["mask_valid"])
        from relearn.adapters import build_dataset
        for model in ("hvq_trans", "dinomaly_small", "u_cat"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "유효 마스크"):
                build_dataset(model, self.manifest(), "val")[2]

    def test_class_map_rejects_missing_or_remapped_classes(self):
        data = json.loads(self.map_path.read_text())
        data["classes"].pop()
        self.map_path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            ClassMap(self.map_path)

    def test_normal_name_can_repeat_across_distinct_items(self):
        data = json.loads(self.map_path.read_text())
        for entry in data["classes"]:
            if entry["kind"] == "normal":
                entry["name"] = "normal"
        self.map_path.write_text(json.dumps(data))
        self.assertEqual(len(ClassMap(self.map_path).normal_ids), 30)

    def test_unknown_labels_fail_before_batch_use(self):
        self.rows[10]["image_label"] = "UNKNOWN"
        self.write()
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            CSVImages(self.manifest(), "val")[0]

    def test_evaluation_augmentation_forbidden(self):
        with self.assertRaisesRegex(ValueError, "Train"):
            CSVImages(self.manifest(), "val", augment=True)

    def test_crop_that_discards_defect_rejected(self):
        for row in self.rows:
            row["roi_xyxy"] = "[0,0,20,20]"
        self.write()
        self.config["crop_to_roi"] = True
        with self.assertRaisesRegex(ValueError, "切|잘라냅니다"):
            CSVImages(self.manifest(), "val", "anomaly")[2]

    def test_fixed_eval_repeated_and_common_train_augmentation(self):
        ds = CSVImages(self.manifest(), "test")
        np.testing.assert_array_equal(ds[0]["image"], ds[0]["image"])
        a = CSVImages(self.manifest(), "train", "anomaly", augment=True)
        b = CSVImages(self.manifest(), "train", "anomaly", augment=True)
        np.testing.assert_array_equal(a[0]["image"], b[0]["image"])
        a.set_epoch(1)
        self.assertFalse(np.array_equal(a[0]["image"], b[0]["image"]))


class AdapterTests(CSVFixture):
    def test_train_shuffle_is_reproducible(self):
        from relearn.adapters import build_loader
        first = build_loader("yolo11s", self.manifest(), "train", batch_size=4)
        second = build_loader("yolo11s", self.manifest(), "train", batch_size=4)
        self.assertEqual(next(iter(first))["im_file"], next(iter(second))["im_file"])

    def test_all_six_adapter_batches(self):
        from relearn.adapters import MODEL_NAMES, build_loader
        for model in MODEL_NAMES:
            for split in ("train", "val", "test"):
                with self.subTest(model=model, split=split):
                    loader = build_loader(model, self.manifest(), split, batch_size=2)
                    batch = next(iter(loader))
                    self.assertIsNotNone(batch)
                    if model in ("hvq_trans", "u_cat"):
                        self.assertEqual(tuple(batch["image"].shape), (2, 3, 448, 448))
                    if model == "yolo11s":
                        self.assertEqual(str(batch["img"].dtype), "torch.uint8")
                        self.assertTrue((batch["cls"] == 111).all() or split == "train")
                    if model == "rtdetrv2_s":
                        self.assertEqual(tuple(batch[0].shape), (2, 3, 448, 448))
                        self.assertTrue(set(batch[1][0]["labels"].tolist()) <= {0, 111})

    def test_worker_spawn_picklability(self):
        from relearn.adapters import build_loader
        loader = build_loader("dinomaly_small", self.manifest(), "val", batch_size=2, workers=1)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch[0].shape), (2, 3, 448, 448))


if __name__ == "__main__":
    unittest.main()
