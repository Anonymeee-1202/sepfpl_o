import os
import pickle
import math
import random
from collections import defaultdict

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import read_json, write_json
from utils.logger import get_global_logger, get_logger


def _get_logger():
    return get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)


# @DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        if cfg.DATASET.USERS >= 20:
            repeat_rate = 0.1
        else:
            repeat_rate = 0

        if cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                num_users=cfg.DATASET.USERS,
                                                                is_iid=cfg.DATASET.IID,
                                                                repeat_rate=repeat_rate)
        elif not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                        num_users=cfg.DATASET.USERS,
                                                                        is_iid=cfg.DATASET.IID,
                                                                        repeat_rate=repeat_rate)
        federated_test_x = self.generate_federated_dataset(test, num_shots=cfg.DATASET.NUM_SHOTS,
                                                            num_users=cfg.DATASET.USERS,
                                                            is_iid=cfg.DATASET.IID,
                                                            repeat_rate=repeat_rate)

        super().__init__(total_train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        logger = _get_logger()
        logger.info("Splitting trainval into %.0f%% train and %.0f%% val", p_trn * 100, p_val * 100)
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        logger = _get_logger()
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        logger.info("Saved split to %s", filepath)

    @staticmethod
    def read_split(filepath, path_prefix):
        logger = _get_logger()
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        logger.info("Reading split from %s", filepath)
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        logger = _get_logger()
        logger.info("SUBSAMPLE %s CLASSES!", subsample.upper())
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    # label=item.label,
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output