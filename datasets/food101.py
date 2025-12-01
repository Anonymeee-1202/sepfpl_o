import os
import pickle
import random

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum


# @DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        federated_train_x, federated_test_x = self.prepare_federated_data(
            train, test, cfg, train_sample_ratio=0.05, test_sample_ratio=0.05
        )

        super().__init__(total_train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x)