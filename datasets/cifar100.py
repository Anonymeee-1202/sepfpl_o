import os
import random
import numpy as np

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from Dassl.dassl.data.datasets import DatasetBase
from datasplit import partition_data, load_cifar100_data

# @DATASET_REGISTRY.register()
class Cifar100():
    dataset_dir = "cifar-100"
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.num_classes = 100

        # ========== 按类别内样本比例采样（保持类别集合不变）==========
        sample_ratio = getattr(cfg.DATASET, 'CIFAR100_SAMPLE_RATIO', 1.0)
        if sample_ratio is None:
            sample_ratio = 1.0
        if sample_ratio <= 0 or sample_ratio > 1:
            sample_ratio = 1.0
        
        # 如果需要进行采样，先加载数据并采样
        if sample_ratio < 1.0:
            # 加载原始数据
            X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar100_data(self.dataset_dir)
            
            # 按类别进行采样
            rng = random.Random(getattr(cfg, 'SEED', 1))
            def per_class_downsample(data_list, labels, images):
                """按类别对数据进行下采样，保持类别集合不变"""
                by_class = {}
                for i, item in enumerate(data_list):
                    classname = item.classname
                    by_class.setdefault(classname, []).append((i, item, labels[i], images[i]))
                
                downsampled_data = []
                downsampled_labels = []
                downsampled_images = []
                
                for cname, items in by_class.items():
                    n_keep = max(1, int(round(len(items) * sample_ratio)))
                    if len(items) <= n_keep:
                        selected = items
                    else:
                        selected = rng.sample(items, n_keep)
                    
                    for idx, item, label, image in selected:
                        downsampled_data.append(item)
                        downsampled_labels.append(label)
                        downsampled_images.append(image)
                
                # 返回采样后的数据（索引会自动从0开始，因为partition_data会基于这些数据创建新索引）
                return downsampled_data, np.array(downsampled_labels), np.array(downsampled_images)
            
            # 对训练集和测试集进行采样
            data_train, y_train, X_train = per_class_downsample(data_train, y_train, X_train)
            data_test, y_test, X_test = per_class_downsample(data_test, y_test, X_test)
            
            # 将采样后的数据传递给 partition_data
            data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test = partition_data(
                'cifar100', self.dataset_dir, cfg.DATASET.PARTITION, cfg.DATASET.USERS, beta=cfg.DATASET.BETA,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                data_train=data_train, data_test=data_test, lab2cname=lab2cname, classnames=classnames)
        else:
            # 不进行采样，使用原始方式
            data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test = partition_data(
                'cifar100', self.dataset_dir, cfg.DATASET.PARTITION, cfg.DATASET.USERS, beta=cfg.DATASET.BETA)

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]
        for net_id in range(cfg.DATASET.USERS):
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            for sample in range(len(dataidxs_train)):
                federated_train_x[net_id].append(data_train[dataidxs_train[sample]])
            for sample in range(len(dataidxs_test)):
                federated_test_x[net_id].append(data_test[dataidxs_test[sample]])

        self.federated_train_x = federated_train_x
        self.federated_test_x = federated_test_x
        self.lab2cname = lab2cname
        self.classnames = classnames


