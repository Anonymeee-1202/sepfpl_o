import os
import pickle
import random

from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):
    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # ========== 按类别内样本比例采样（保持类别集合不变） ==========
        rng = random.Random(getattr(cfg, 'SEED', 1))
        # train = self.per_class_downsample(train, 0.5, rng)
        test = self.per_class_downsample(test, 0.5, rng)

        federated_train_x, federated_test_x = self.prepare_federated_data(
            train, test, cfg, train_sample_ratio=None, test_sample_ratio=None
        )

        super().__init__(total_train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x)