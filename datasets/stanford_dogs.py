import os
import random

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum


# @DATASET_REGISTRY.register()
class StanfordDogs(DatasetBase):
    """
    Stanford Dogs Dataset 类定义。
    
    Stanford Dogs Dataset 包含 120 个狗品种类别，通常有两种组织方式：
    1. 有 train_list.txt 和 test_list.txt 分割文件（类似 Oxford Pets）
    2. 按类别文件夹组织（类似 DTD）
    
    本实现优先使用分割文件，如果不存在则使用按类别文件夹的方式。
    """
    dataset_dir = "stanford_dogs"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Images")
        self.anno_dir = os.path.join(self.dataset_dir, "Annotation")
        self.split_path = os.path.join(self.dataset_dir, "split_StanfordDogs.json")

        # 检查是否存在预定义的分割文件
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            # 尝试使用分割文件（train_list.txt, test_list.txt）
            train_list_path = os.path.join(self.dataset_dir, "train_list.txt")
            test_list_path = os.path.join(self.dataset_dir, "test_list.txt")
            
            if os.path.exists(train_list_path) and os.path.exists(test_list_path):
                # 使用分割文件方式
                trainval = self.read_data(split_file="train_list.txt")
                test = self.read_data(split_file="test_list.txt")
                train, val = self.split_trainval(trainval)
                self.save_split(train, val, test, self.split_path, self.image_dir)
            else:
                # 使用按类别文件夹方式（类似 DTD）
                train, val, test = self.read_and_split_data(self.image_dir)
                self.save_split(train, val, test, self.split_path, self.image_dir)

        # ========== 按类别内样本比例采样（保持类别集合不变） ==========
        rng = random.Random(getattr(cfg, 'SEED', 1))
        # train = self.per_class_downsample(train, 0.5, rng)
        test = self.per_class_downsample(test, 0.5, rng)

        federated_train_x, federated_test_x = self.prepare_federated_data(
            train, test, cfg, train_sample_ratio=None, test_sample_ratio=None
        )

        super().__init__(
            total_train_x=train,
            federated_train_x=federated_train_x,
            federated_test_x=federated_test_x
        )

    def read_data(self, split_file):
        """
        从分割文件中读取数据。
        
        参数:
            split_file (str): 分割文件名，如 "train_list.txt" 或 "test_list.txt"
        
        返回:
            items (list): Datum 对象列表
        
        分割文件格式通常为：
            n02085620-Chihuahua/n02085620_10074.jpg 1
            其中第一列是相对路径，第二列是类别标签（1-based）
        """
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        if not os.path.exists(filepath):
            return items

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 处理不同的文件格式
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                # 获取图像路径和标签
                imname = parts[0]
                label = int(parts[1]) - 1  # 转换为 0-based 索引
                
                # 构建完整路径
                if not imname.endswith(('.jpg', '.jpeg', '.png')):
                    imname += ".jpg"
                
                impath = os.path.join(self.image_dir, imname)
                
                # 如果图像不存在，尝试其他可能的路径
                if not os.path.exists(impath):
                    # 尝试在 Images 目录下直接查找
                    alt_path = os.path.join(self.image_dir, os.path.basename(imname))
                    if os.path.exists(alt_path):
                        impath = alt_path
                    else:
                        continue
                
                # 从路径中提取类别名称
                # 例如：n02085620-Chihuahua/n02085620_10074.jpg -> Chihuahua
                classname = os.path.dirname(imname)
                if "-" in classname:
                    # 格式：n02085620-Chihuahua -> Chihuahua
                    classname = classname.split("-", 1)[1]
                else:
                    # 如果没有分隔符，使用目录名
                    classname = os.path.basename(os.path.dirname(impath))
                
                classname = classname.lower().replace("_", " ")
                
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


