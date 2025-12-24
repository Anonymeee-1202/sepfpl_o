import os
import random
import numpy as np
from collections import defaultdict
from prettytable import PrettyTable

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from Dassl.dassl.data.datasets import DatasetBase
from datasplit import partition_data, load_cifar100_data
from utils.logger import require_global_logger

# @DATASET_REGISTRY.register()
class Cifar100():
    dataset_dir = "cifar-100"
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.num_classes = 100

        # ========== 按类别内样本比例采样（保持类别集合不变）==========
        # 手动设置采样比例，可根据需要修改此值
        sample_ratio = 0.4
        
        # 加载原始数据
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar100_data(self.dataset_dir)
        
        # 按类别进行采样
        rng = random.Random(getattr(cfg, 'SEED', 1))
        def per_class_downsample(data_list, labels, images):
            """按类别对数据进行下采样，保持类别集合不变"""
            by_class = {}
            for i, item in enumerate(data_list):
                by_class.setdefault(item.classname, []).append((item, labels[i], images[i]))
            
            downsampled_data, downsampled_labels, downsampled_images = [], [], []
            for items in by_class.values():
                n_keep = max(1, int(round(len(items) * sample_ratio)))
                selected = items if len(items) <= n_keep else rng.sample(items, n_keep)
                for item, label, image in selected:
                    downsampled_data.append(item)
                    downsampled_labels.append(label)
                    downsampled_images.append(image)
            
            return downsampled_data, np.array(downsampled_labels), np.array(downsampled_images)
        
        # 对训练集和测试集进行采样
        data_train, y_train, X_train = per_class_downsample(data_train, y_train, X_train)
        data_test, y_test, X_test = per_class_downsample(data_test, y_test, X_test)
        
        # 将采样后的数据传递给 partition_data
        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test = partition_data(
            'cifar100', self.dataset_dir, cfg.DATASET.PARTITION, cfg.DATASET.USERS, beta=cfg.DATASET.BETA,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            data_train=data_train, data_test=data_test, lab2cname=lab2cname, classnames=classnames)

        # 构建联邦训练和测试数据
        federated_train_x = [[data_train[idx] for idx in net_dataidx_map_train[net_id]] 
                             for net_id in range(cfg.DATASET.USERS)]
        federated_test_x = [[data_test[idx] for idx in net_dataidx_map_test[net_id]] 
                            for net_id in range(cfg.DATASET.USERS)]

        self.federated_train_x = federated_train_x
        self.federated_test_x = federated_test_x
        self.lab2cname = lab2cname
        self.classnames = classnames
        
        # 输出数据划分结果表格
        logger = require_global_logger()
        if federated_train_x:
            # 统计每个客户端的类别分布
            user_class_dict = defaultdict(set)
            user_class_samples = defaultdict(lambda: defaultdict(int))
            
            for net_id in range(cfg.DATASET.USERS):
                for item in federated_train_x[net_id]:
                    # CIFAR100_truncated 数据项的 label 属性
                    label = item.label
                    user_class_dict[net_id].add(label)
                    user_class_samples[net_id][label] += 1
            
            # 创建表格
            table = PrettyTable()
            table.field_names = ["用户ID", "类别数", "类别详情 (类别ID(样本数))", "总样本数"]
            table.align = "l"
            
            for idx in sorted(range(cfg.DATASET.USERS)):
                classes = sorted(user_class_dict[idx])
                # 构建类别详情字符串，格式：类别ID(样本数)
                class_details = []
                for cls in classes:
                    sample_count = user_class_samples[idx].get(cls, 0)
                    class_details.append(f"{cls}({sample_count})")
                classes_str = ", ".join(class_details)
                # 如果类别太多，截断显示
                if len(classes_str) > 100:
                    classes_str = classes_str[:97] + "..."
                
                total_samples = len(federated_train_x[idx])
                table.add_row([
                    f"User {idx}",
                    len(classes),
                    classes_str,
                    total_samples
                ])
            
            logger.info(f"\nCIFAR-100 训练数据划分结果:\n{table}")


