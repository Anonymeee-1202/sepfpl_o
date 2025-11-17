import os
import pickle
import random

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


# @DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # ========== 按类别内样本比例采样（保持类别集合不变） ==========
        sample_ratio = getattr(cfg.DATASET, 'FOOD101_SAMPLE_RATIO', 1.0)
        if sample_ratio is None:
            sample_ratio = 1.0
        if sample_ratio <= 0 or sample_ratio > 1:
            sample_ratio = 1.0
        if sample_ratio < 1.0:
            rng = random.Random(getattr(cfg, 'SEED', 1))
            def per_class_downsample(data_list):
                by_class = {}
                for d in data_list:
                    by_class.setdefault(d.classname, []).append(d)
                downsampled = []
                for cname, items in by_class.items():
                    n_keep = max(1, int(round(len(items) * sample_ratio)))
                    if len(items) <= n_keep:
                        downsampled.extend(items)
                    else:
                        downsampled.extend(rng.sample(items, n_keep))
                return downsampled
            train = per_class_downsample(train)
            test = per_class_downsample(test)

        if cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                num_users=cfg.DATASET.USERS,
                                                                is_iid=cfg.DATASET.IID,
                                                                repeat_rate=0)
        elif not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                        num_users=cfg.DATASET.USERS,
                                                                        is_iid=cfg.DATASET.IID,
                                                                        repeat_rate=0)
        federated_test_x = self.generate_federated_dataset(test, num_shots=cfg.DATASET.NUM_SHOTS,
                                                            num_users=cfg.DATASET.USERS,
                                                            is_iid=cfg.DATASET.IID,
                                                            repeat_rate=0)

        super().__init__(total_train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x)