import os
import pickle
import math
import random
from collections import defaultdict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum


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

        # ========== 按类别内样本比例采样（保持类别集合不变） ==========
        rng = random.Random(getattr(cfg, 'SEED', 1))
        # train = self.per_class_downsample(train, 0.25, rng)
        test = self.per_class_downsample(test, 0.25, rng)

        federated_train_x, federated_test_x = self.prepare_federated_data(
            train, test, cfg, train_sample_ratio=None, test_sample_ratio=None
        )

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