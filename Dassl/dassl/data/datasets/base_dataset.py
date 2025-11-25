import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown

from Dassl.dassl.utils import check_isfile
from utils.logger import require_global_logger


def _get_dataset_logger():
    return require_global_logger()


class Datum:
    """
    数据实例类，定义了数据的基本属性。

    参数:
        impath (str): 图片路径。
        label (int): 类别标签 (ID)。
        domain (int): 领域标签 (ID)。
        classname (str): 类别名称。
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath), f"File not found: {impath}"

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """
    统一的数据集基类，适用于：
    1) 领域自适应 (Domain Adaptation)
    2) 领域泛化 (Domain Generalization)
    3) 半监督学习 (Semi-supervised Learning)
    4) 联邦学习 (Federated Learning)
    """

    dataset_dir = ""  # 数据集存储目录
    domains = []      # 所有领域的名称列表

    def __init__(self, total_train_x=None, federated_train_x=None, federated_test_x=None):
        self._federated_train_x = federated_train_x # 联邦学习：有标签训练数据
        self._federated_test_x = federated_test_x   # 联邦学习：有标签测试数据
        # 基于全量训练数据计算类别信息
        self._num_classes = self.get_num_classes(total_train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(total_train_x)

    @property
    def federated_train_x(self):
        return self._federated_train_x

    @property
    def federated_test_x(self):
        return self._federated_test_x

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """
        统计类别总数。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        返回:
            int: 类别总数。
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """
        获取 标签到类别名 的映射字典。

        参数:
            data_source (list): 包含 Datum 对象的列表。
        返回:
            mapping (dict): {label: classname}
            classnames (list): 按 label 排序的 classname 列表
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        """检查输入源域和目标域是否合法"""
        assert len(source_domains) > 0, "source_domains (list) 为空"
        assert len(target_domains) > 0, "target_domains (list) 为空"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "输入领域必须属于 {}, "
                    "但收到了 [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        """
        下载并解压数据集。
        """
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError("目前仅支持从 Google Drive 下载")

        logger = _get_dataset_logger()
        logger.info("正在解压文件 ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError("不支持的文件格式")

        logger.info("文件已解压至 {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """
        生成少样本 (Few-shot) 数据集 (通常用于集中式训练集)。
        这在每个类仅包含少量图片的少样本学习设置中很有用。

        参数:
            data_sources: 可变参数，每个元素是一个包含 Datum 对象的列表。
            num_shots (int): 每个类采样的实例数量 (-1 表示使用所有数据)。
            repeat (bool): 如果样本不足，是否允许重复采样 (默认: False)。
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        logger = _get_dataset_logger()
        logger.info(f"正在创建 {num_shots}-shot 数据集")

        output = []
        logger.info("data_sources 长度: %d", len(data_sources))

        for data_source in data_sources:
            logger.info("当前 data_source 长度: %d", len(data_source))
            # 按类别将数据分组
            tracker = self.split_dataset_by_label(data_source)
            logger.info("包含类别的数量: %d", len(tracker))
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    # 样本充足，不放回采样
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        # 样本不足且允许重复，有放回采样
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        # 样本不足且不允许重复，取所有样本
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def generate_federated_fewshot_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate=0.0, repeat=False
    ):
        """
        生成联邦少样本 (Federated Few-shot) 数据集 (通常用于联邦训练集)。

        参数:
            data_sources: 数据源列表。
            num_shots (int): 每个类采样的实例数量。
            num_users (int): 用户(Client)数量。
            is_iid (bool): 是否独立同分布 (IID)。如果为 True，每个用户拥有所有类。
            repeat_rate (float): 类别重叠率 (0.0 - 1.0)。控制不同用户间共享类别的比例 (用于 Non-IID 设置)。
            repeat (bool): 采样时是否允许重复。

        返回:
            output_dict (dict): {user_id: [Datum_list]}，每个用户的训练数据列表。
        """
        logger = _get_dataset_logger()
        logger.info(f"正在创建 {num_shots}-shot 联邦数据集")
        output_dict = defaultdict(list)

        # 情况 1: 不进行少样本采样 (使用全部数据)，简单分配
        if num_shots < 1:
            for idx in range(num_users):
                if len(data_sources) == 1:
                    output_dict[idx] = data_sources[0]
                output_dict[idx].append(data_sources)

        # 情况 2: 执行联邦少样本采样逻辑
        else:
            user_class_dict = defaultdict(list) # 存储每个用户被分配到的类别ID列表
            class_num = self.get_num_classes(data_sources[0])
            logger.info("总类别数: %d", class_num)
            
            # 基础分配：每个用户平均分到的类别数
            class_per_user = int(round(class_num / num_users))
            
            # 生成随机打乱的类别列表
            class_list = list(range(0, class_num))
            random.seed(2023)
            random.shuffle(class_list)

            # --- 逻辑块：处理 Non-IID 下的类别重叠 (Repeat Rate) ---
            if repeat_rate > 0:
                repeat_num = int(repeat_rate * class_num) # 重复(共享)的类别数量
                class_repeat_list = class_list[0:repeat_num]
                class_norepeat_list = class_list[repeat_num:class_num]
                
                # 扣除重复类后，每个用户分配到的"独占"类别数
                class_per_user = int(round((class_num - repeat_num) / num_users))
                
                # Fold 计算：用于将用户分组，组内共享特定的重复类
                fold = int(num_users / num_shots) if num_shots > 0 else 0 
                logger.info("重复类别数: %d", repeat_num)
                logger.info("Fold 数: %d", fold)

                if fold > 0:
                    client_idx_fold = defaultdict(list)
                    client_per_fold = int(round(num_users / fold))
                    repeat_per_fold = int(round(repeat_num / fold))
                    
                    client_list = list(range(0, num_users))
                    random.shuffle(client_list)
                    
                    # 将用户分配到不同的 fold
                    for i in range(fold):
                        client_idx_fold[i] = client_list[i * client_per_fold : min((i + 1) * client_per_fold, num_users)]

            for data_source in data_sources:
                tracker = self.split_dataset_by_label(data_source)

                # --- 步骤 1: 确定每个用户拥有的类别列表 (user_class_dict) ---
                for idx in range(num_users):
                    if is_iid:
                        # IID 设置：用户拥有所有类别
                        user_class_dict[idx] = list(range(0, class_num))
                    else:
                        # Non-IID 设置
                        if repeat_rate == 0.0:
                            # 无重叠：严格切分类别列表
                            if idx == num_users - 1:
                                user_class_dict[idx] = class_list[idx * class_per_user : class_num]
                            else:
                                user_class_dict[idx] = class_list[idx * class_per_user : (idx + 1) * class_per_user]
                        else:
                            # 有重叠 (repeat_rate > 0)
                            user_class_dict[idx] = []
                            # 1.1 添加共享(重复)部分
                            if fold > 0:
                                for k, v in client_idx_fold.items():
                                    if idx in v: # 找到当前用户所在的 fold
                                        if k == len(client_idx_fold) - 1:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold : repeat_num])
                                        else:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold : (k + 1) * repeat_per_fold])
                            else:
                                user_class_dict[idx].extend(class_repeat_list)
                            
                            logger.info("User %d repeat part: %s", idx, user_class_dict[idx])

                            # 1.2 添加私有(不重复)部分
                            if idx == num_users - 1:
                                segment = class_norepeat_list[idx * class_per_user : class_num - repeat_num]
                                user_class_dict[idx].extend(segment)
                            else:
                                segment = class_norepeat_list[idx * class_per_user : (idx + 1) * class_per_user]
                                user_class_dict[idx].extend(segment)
                            
                            logger.info("User %d non-repeat part: %s", idx, segment)

                    logger.info("User %d total classes: %s", idx, user_class_dict[idx])

                    # --- 步骤 2: 根据类别列表进行数据采样 ---
                    dataset = []
                    for label, items in tracker.items():
                        if label in user_class_dict[idx]: # 仅处理用户拥有的类别
                            
                            if repeat_rate == 0.0:
                                # 标准采样
                                if len(items) >= num_shots:
                                    sampled_items = random.sample(items, num_shots)
                                else:
                                    if repeat:
                                        sampled_items = random.choices(items, k=num_shots)
                                    else:
                                        sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                # 针对 Repeat Rate > 0 的特殊采样逻辑
                                # 如果当前类是"共享类"，为了避免所有用户都取同样的样本或样本不够分，
                                # 这里尝试将 num_shots 均分给用户 (tmp_num_shots)
                                if label in class_repeat_list:
                                    tmp_num_shots = int(num_shots / num_users) if int(num_shots / num_users) > 0 else 1
                                    sampled_items = random.sample(items, tmp_num_shots) # 注意：这里可能会导致所有用户取到的样本很少
                                else:
                                    sampled_items = random.sample(items, num_shots)
                                dataset.extend(sampled_items)

                    output_dict[idx] = dataset
                    logger.info("idx: %s, output_dict_len: %d", idx, len(output_dict[idx]))

        return output_dict

    def generate_federated_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate=0.0, repeat=False
    ):
        """
        生成联邦数据集 (通常用于联邦 Baseline 训练集)。
        与 few-shot 版本不同，这里试图模拟每个客户端拥有其类别的全部或特定比例的数据。

        参数:
            num_shots (int): 虽然命名为 shots，但在 baseline 中可能用于控制 fold 分组逻辑。
            ... (其他参数同上)
        """
        logger = _get_dataset_logger()
        logger.info(f"正在创建 Baseline 联邦数据集")
        output_dict = defaultdict(list)
        user_class_dict = defaultdict(list)
        sample_per_user = defaultdict(int) # 每个类别分给每个用户的样本数
        sample_order = defaultdict(list)   # 记录每个类别样本的分配顺序索引

        class_num = self.get_num_classes(data_sources[0])
        logger.info("类别总数: %d", class_num)
        
        class_per_user = int(round(class_num / num_users))
        class_list = list(range(0, class_num))
        random.seed(2023)
        random.shuffle(class_list)

        # --- 逻辑块：准备 Non-IID 参数 (Repeat Rate & Fold) ---
        if repeat_rate > 0:
            repeat_num = int(repeat_rate * class_num)
            class_repeat_list = class_list[0:repeat_num]
            class_norepeat_list = class_list[repeat_num:class_num]
            class_per_user = int(round((class_num - repeat_num) / num_users))
            
            fold = int(num_users / num_shots) if num_shots > 0 else 0
            logger.info("repeat_num: %d", repeat_num)
            logger.info("fold: %d", fold)

            if fold > 0:
                client_idx_fold = defaultdict(list)
                client_per_fold = int(round(num_users / fold))
                repeat_per_fold = int(round(repeat_num / fold))
                client_list = list(range(0, num_users))
                random.shuffle(client_list)
                for i in range(fold):
                    client_idx_fold[i] = client_list[i * client_per_fold : min((i + 1) * client_per_fold, num_users)]

        # --- 步骤 1: 预计算每个类别的样本分配策略 ---
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            for label, items in tracker.items():
                sample_order[label] = list(range(0, len(items)))
                # 默认将该类样本平均分给所有用户 (如果是 IID 或 共享类)
                sample_per_user[label] = int(round(len(items) / num_users))
                random.shuffle(sample_order[label])
                
                # 如果使用了 fold 分组，共享类的样本分配量需要调整
                if repeat_rate > 0 and 'fold' in locals() and fold > 0:
                     # 此时共享该类的用户数减少了，每个用户分到的样本变多
                    sample_per_user[label] = int(round(len(items) / (num_users / fold)))

            # --- 步骤 2: 为每个用户分配类别和数据 ---
            for idx in range(num_users):
                # 2.1 确定用户拥有的类别 ID (逻辑同 generate_federated_fewshot_dataset)
                if is_iid:
                    user_class_dict[idx] = list(range(0, class_num))
                else:
                    if repeat_rate == 0.0:
                        if idx == num_users - 1:
                            user_class_dict[idx] = class_list[idx * class_per_user : class_num]
                        else:
                            user_class_dict[idx] = class_list[idx * class_per_user : (idx + 1) * class_per_user]
                    else:
                        user_class_dict[idx] = []
                        if fold > 0:
                            for k, v in client_idx_fold.items():
                                if idx in v:
                                    if k == len(client_idx_fold) - 1:
                                        user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold : repeat_num])
                                    else:
                                        user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold : (k + 1) * repeat_per_fold])
                        else:
                            user_class_dict[idx].extend(class_repeat_list)

                        logger.info("User %d repeat part: %s", idx, user_class_dict[idx])

                        if idx == num_users - 1:
                            segment = class_norepeat_list[idx * class_per_user : class_num - repeat_num]
                            user_class_dict[idx].extend(segment)
                        else:
                            segment = class_norepeat_list[idx * class_per_user : (idx + 1) * class_per_user]
                            user_class_dict[idx].extend(segment)
                        
                        logger.info("User %d non-repeat part: %s", idx, segment)

                logger.info("User %d total classes: %s", idx, user_class_dict[idx])

                dataset = []

                # 2.2 根据 sample_order 和 sample_per_user 进行切片分配
                for label, items in tracker.items():
                    if label in user_class_dict[idx]:
                        if is_iid:
                            # IID: 按照预先计算的顺序切片，确保每个用户拿到不重复的切片
                            sampled_items = []
                            for k, v in enumerate(items):
                                start_idx = idx * sample_per_user[label]
                                end_idx = min((idx + 1) * sample_per_user[label], len(items))
                                if k in sample_order[label][start_idx : end_idx]:
                                    sampled_items.append(v)
                            dataset.extend(sampled_items)
                        else:
                            # Non-IID
                            if repeat_rate == 0.0:
                                # 独占该类，拿走所有数据
                                sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                # 有重叠
                                if label in user_class_dict[idx][0:repeat_num]:
                                    # 是共享类：只拿走属于该用户切片的部分
                                    sampled_items = []
                                    for k, v in enumerate(items):
                                        start_idx = idx * sample_per_user[label]
                                        end_idx = min((idx + 1) * sample_per_user[label], len(items))
                                        if k in sample_order[label][start_idx : end_idx]:
                                            sampled_items.append(v)
                                    dataset.extend(sampled_items)
                                else:
                                    # 是独占类：拿走所有数据
                                    sampled_items = items
                                    dataset.extend(sampled_items)

                output_dict[idx] = dataset
                logger.info("idx: %s, output_dict_len: %d", idx, len(output_dict[idx]))

        return output_dict

    def split_dataset_by_label(self, data_source):
        """
        按 Label 拆分数据集。
        将 Datum 对象列表转换为以 Label 为键的字典。

        参数:
            data_source (list): Datum 对象列表。
        返回:
            output (dict): {label: [Datum_list]}
        """
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output

    def split_dataset_by_domain(self, data_source):
        """
        按 Domain 拆分数据集。
        将 Datum 对象列表转换为以 Domain 为键的字典。

        参数:
            data_source (list): Datum 对象列表。
        返回:
            output (dict): {domain: [Datum_list]}
        """
        output = defaultdict(list)
        for item in data_source:
            output[item.domain].append(item)
        return output