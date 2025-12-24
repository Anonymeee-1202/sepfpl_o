import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
from prettytable import PrettyTable

from Dassl.dassl.utils import check_isfile, read_json, write_json, listdir_nohidden
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

        logger.info(f"文件已解压至 {osp.dirname(dst)}")

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
        logger.info(f"data_sources 长度: {len(data_sources)}")

        for data_source in data_sources:
            logger.info(f"当前 data_source 长度: {len(data_source)}")
            # 按类别将数据分组
            tracker = self.split_dataset_by_label(data_source)
            logger.info(f"包含类别的数量: {len(tracker)}")
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
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate=0.0, repeat=False, dataset_type="训练"
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
            logger.info(f"总类别数: {class_num}")
            
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
                logger.info(f"重复类别数: {repeat_num}")
                logger.info(f"Fold 数: {fold}")

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

        # 输出表格
        if output_dict:
            table = PrettyTable()
            table.field_names = ["用户ID", "类别数", "类别列表", "样本数"]
            table.align = "l"
            for idx in sorted(output_dict.keys()):
                classes = sorted(user_class_dict[idx]) if idx in user_class_dict else []
                classes_str = str(classes) if len(classes) <= 20 else str(classes[:20]) + "..."
                table.add_row([
                    f"User {idx}",
                    len(classes),
                    classes_str,
                    len(output_dict[idx])
                ])
            logger.info(f"\n{dataset_type}数据划分 ({num_shots}-shot):\n{table}")

        return output_dict

    def generate_federated_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate=0.0, repeat=False, dataset_type="训练"
    ):
        """
        生成联邦数据集 (通常用于联邦 Baseline 训练集)。
        与 few-shot 版本不同，这里试图模拟每个客户端拥有其类别的全部或特定比例的数据。

        参数:
            data_sources: 可变参数，每个元素是一个包含 Datum 对象的列表。
            num_shots (int): 用于控制 fold 分组逻辑。fold = num_users / num_shots。
            num_users (int): 用户(Client)数量。
            is_iid (bool): 是否独立同分布 (IID)。如果为 True，每个用户拥有所有类。
            repeat_rate (float): 类别重叠率 (0.0 - 1.0)。控制不同用户间共享类别的比例 (用于 Non-IID 设置)。
            repeat (bool): 未使用，保留以保持接口一致性。

        返回:
            output_dict (dict): {user_id: [Datum_list]}，每个用户的训练数据列表。
        """
        logger = _get_dataset_logger()
        logger.info(f"正在创建 Baseline 联邦{dataset_type}数据集")
        output_dict = defaultdict(list)
        user_class_dict = defaultdict(list)
        user_class_samples = defaultdict(dict)  # 记录每个用户每个类别的样本数 {user_id: {label: count}}
        sample_per_user = defaultdict(int)  # 每个类别分给每个用户的样本数
        sample_order = defaultdict(list)    # 记录每个类别样本的分配顺序索引

        class_num = self.get_num_classes(data_sources[0])
        
        # 特殊处理：当 num_users > class_num 时，将用户分成 class_num 个组，每个组对应一个类别，组内 IID 分配
        if num_users > class_num:
            logger.info(f"用户数 ({num_users}) 大于类别数 ({class_num})，将用户分成 {class_num} 个组，每组对应一个类别")
            
            # 将用户分成 class_num 个组
            users_per_class = num_users // class_num  # 每个类别组的基础用户数
            extra_users = num_users % class_num  # 需要额外分配的用户数
            
            class_list = list(range(0, class_num))
            random.seed(2023)
            random.shuffle(class_list)
            
            user_list = list(range(0, num_users))
            random.shuffle(user_list)
            
            # 为每个类别分配用户组
            class_to_users = defaultdict(list)  # {class_id: [user_ids]}
            user_idx = 0
            for class_idx, class_id in enumerate(class_list):
                # 计算这个类别组应该有多少用户
                num_users_for_class = users_per_class + (1 if class_idx < extra_users else 0)
                # 分配用户到这个类别组
                class_to_users[class_id] = user_list[user_idx:user_idx + num_users_for_class]
                user_idx += num_users_for_class
            
            # 为每个用户分配类别和数据
            for data_source in data_sources:
                tracker = self.split_dataset_by_label(data_source)
                
                for class_id, user_ids in class_to_users.items():
                    if class_id not in tracker:
                        continue
                    
                    items = tracker[class_id]
                    num_users_in_group = len(user_ids)
                    
                    # 组内 IID 分配：将该类别的数据平均分配给组内用户
                    if class_id not in sample_order:
                        sample_order[class_id] = list(range(0, len(items)))
                        random.shuffle(sample_order[class_id])
                    samples_per_user = len(items) // num_users_in_group
                    
                    for group_user_idx, user_id in enumerate(user_ids):
                        if user_id not in user_class_dict:
                            user_class_dict[user_id] = []
                        if class_id not in user_class_dict[user_id]:
                            user_class_dict[user_id].append(class_id)
                        
                        # 计算该用户应该分到的数据切片
                        start_idx = group_user_idx * samples_per_user
                        if group_user_idx == num_users_in_group - 1:
                            # 最后一个用户拿走剩余的所有数据
                            end_idx = len(items)
                        else:
                            end_idx = (group_user_idx + 1) * samples_per_user
                        
                        selected_indices = sample_order[class_id][start_idx:end_idx]
                        sampled_items = [items[i] for i in selected_indices]
                        
                        # 合并多个数据源的数据
                        if user_id not in output_dict:
                            output_dict[user_id] = []
                        output_dict[user_id].extend(sampled_items)
                        
                        # 更新样本计数
                        if user_id not in user_class_samples:
                            user_class_samples[user_id] = {}
                        if class_id not in user_class_samples[user_id]:
                            user_class_samples[user_id][class_id] = 0
                        user_class_samples[user_id][class_id] += len(sampled_items)
            
            # 输出表格
            if output_dict:
                table = PrettyTable()
                table.field_names = ["用户ID", "类别数", "类别详情 (类别ID(样本数))", "总样本数"]
                table.align = "l"
                for idx in sorted(output_dict.keys()):
                    classes = sorted(user_class_dict[idx]) if idx in user_class_dict else []
                    # 构建类别详情字符串，格式：类别ID(样本数)
                    class_details = []
                    for cls in classes:
                        sample_count = user_class_samples[idx].get(cls, 0)
                        class_details.append(f"{cls}({sample_count})")
                    classes_str = ", ".join(class_details)
                    # 如果类别太多，截断显示
                    if len(classes_str) > 100:
                        classes_str = classes_str[:97] + "..."
                    
                    total_samples = len(output_dict[idx])
                    table.add_row([
                        f"User {idx}",
                        len(classes),
                        classes_str,
                        total_samples
                    ])
                logger.info(f"\n{dataset_type}数据划分 (Baseline):\n{table}")
            
            return output_dict
        
        class_per_user = int(round(class_num / num_users))
        class_list = list(range(0, class_num))
        random.seed(2023)
        random.shuffle(class_list)

        # --- 逻辑块：准备 Non-IID 参数 (Repeat Rate & Fold) ---
        fold = 0
        if repeat_rate > 0:
            repeat_num = int(repeat_rate * class_num)
            class_repeat_list = class_list[0:repeat_num]
            class_norepeat_list = class_list[repeat_num:class_num]
            class_per_user = int(round((class_num - repeat_num) / num_users))
            
            fold = int(num_users / num_shots) if num_shots > 0 else 0
            logger.info(f"重复类别数: {repeat_num}")
            logger.info(f"Fold 数: {fold}")

            if fold > 0:
                client_idx_fold = defaultdict(list)
                client_per_fold = int(round(num_users / fold))
                repeat_per_fold = int(round(repeat_num / fold))
                client_list = list(range(0, num_users))
                random.shuffle(client_list)

                # 将用户分配到不同的 fold
                for i in range(fold):
                    start_idx = i * client_per_fold
                    end_idx = min((i + 1) * client_per_fold, num_users)
                    client_idx_fold[i] = client_list[start_idx:end_idx]

        # --- 步骤 1: 预计算每个类别的样本分配策略 ---
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)

            for label, items in tracker.items():
                sample_order[label] = list(range(0, len(items)))
                # 默认将该类样本平均分给所有用户 (如果是 IID 或共享类)
                sample_per_user[label] = int(round(len(items) / num_users))
                random.shuffle(sample_order[label])
                
                # 如果使用了 fold 分组，共享类的样本分配量需要调整
                     # 此时共享该类的用户数减少了，每个用户分到的样本变多
                if repeat_rate > 0 and fold > 0:
                    sample_per_user[label] = int(round(len(items) / (num_users / fold)))

            # --- 步骤 2: 为每个用户分配类别（均匀分配，差值不超过1）---
            # 首先为所有用户分配共享类别（如果有）
            if repeat_rate > 0 and not is_iid:
                # 预先分配共享类别
                for idx in range(num_users):
                    user_class_dict[idx] = []
                    if fold > 0:
                        for k, v in client_idx_fold.items():
                            if idx in v:  # 找到当前用户所在的 fold
                                if k == len(client_idx_fold) - 1:
                                    user_class_dict[idx].extend(
                                        class_repeat_list[k * repeat_per_fold:repeat_num]
                                    )
                                else:
                                    end_idx = (k + 1) * repeat_per_fold
                                    user_class_dict[idx].extend(
                                        class_repeat_list[k * repeat_per_fold:end_idx]
                                    )
                    else:
                        user_class_dict[idx].extend(class_repeat_list)
                
                # 计算每个用户已分配的共享类别数
                shared_class_counts = [len(user_class_dict[idx]) for idx in range(num_users)]
                
                # 计算每个用户应该分配的总类别数（均匀分配，差值不超过1）
                total_classes_to_assign = class_num  # 总类别数
                # 计算平均每个用户应该有的类别数
                avg_classes_per_user = total_classes_to_assign / num_users
                base_classes = int(avg_classes_per_user)  # 基础类别数
                extra_classes = total_classes_to_assign - base_classes * num_users  # 需要额外分配的类别数
                
                # 计算每个用户还需要分配的私有类别数
                private_class_counts = []
                for idx in range(num_users):
                    target_total = base_classes + (1 if idx < extra_classes else 0)
                    needed_private = max(0, target_total - shared_class_counts[idx])
                    private_class_counts.append(needed_private)
                
                # 均匀分配私有类别
                private_class_idx = 0
                for idx in range(num_users):
                    if private_class_counts[idx] > 0:
                        end_idx = private_class_idx + private_class_counts[idx]
                        user_class_dict[idx].extend(
                            class_norepeat_list[private_class_idx:end_idx]
                        )
                        private_class_idx = end_idx
            else:
                # 无重叠或IID情况：均匀分配所有类别
                for idx in range(num_users):
                    if is_iid:
                        user_class_dict[idx] = list(range(0, class_num))
                    else:
                        # 无重叠：均匀切分类别列表
                        # 计算每个用户应该分配的类别数（差值不超过1）
                        avg_classes = class_num / num_users
                        base_classes = int(avg_classes)
                        extra_classes = class_num - base_classes * num_users
                        
                        if idx < extra_classes:
                            num_classes_for_user = base_classes + 1
                        else:
                            num_classes_for_user = base_classes
                        
                        start_idx = sum(
                            base_classes + (1 if i < extra_classes else 0) 
                            for i in range(idx)
                        )
                        end_idx = start_idx + num_classes_for_user
                        user_class_dict[idx] = class_list[start_idx:end_idx]

            # --- 步骤 3: 为每个用户分配数据 ---
            for idx in range(num_users):
                # 2.2 根据 sample_order 和 sample_per_user 进行切片分配
                # 先按 label 收集数据，然后交错排列
                label_data_dict = {}  # {label: [items]}
                
                for label, items in tracker.items():
                    if label not in user_class_dict[idx]:
                        continue

                    if is_iid:
                        # IID: 按照预先计算的顺序切片，确保每个用户拿到不重复的切片
                        sampled_items = []
                        start_idx = idx * sample_per_user[label]
                        end_idx = min((idx + 1) * sample_per_user[label], len(items))
                        selected_indices = sample_order[label][start_idx:end_idx]

                        for k, v in enumerate(items):
                            if k in selected_indices:
                                sampled_items.append(v)
                        label_data_dict[label] = sampled_items
                    else:
                        # Non-IID
                        if repeat_rate == 0.0:
                            # 独占该类，拿走所有数据
                            label_data_dict[label] = items
                        else:
                            # 有重叠
                            if label in class_repeat_list:
                                # 是共享类：只拿走属于该用户切片的部分
                                sampled_items = []
                                start_idx = idx * sample_per_user[label]
                                end_idx = min((idx + 1) * sample_per_user[label], len(items))
                                selected_indices = sample_order[label][start_idx:end_idx]

                                for k, v in enumerate(items):
                                    if k in selected_indices:
                                        sampled_items.append(v)
                                label_data_dict[label] = sampled_items
                            else:
                                # 是独占类：拿走所有数据
                                label_data_dict[label] = items

                # 记录每个类别的样本数
                for label, items in label_data_dict.items():
                    user_class_samples[idx][label] = len(items)
                
                # 使用 round-robin 方式交错排列不同 label 的数据
                dataset = []
                if label_data_dict:
                    # 获取所有 label 和对应的数据列表
                    labels = list(label_data_dict.keys())
                    data_lists = [label_data_dict[label] for label in labels]
                    
                    # 计算最大长度
                    max_len = max(len(data_list) for data_list in data_lists) if data_lists else 0
                    
                    # Round-robin 交错排列
                    for i in range(max_len):
                        for j, data_list in enumerate(data_lists):
                            if i < len(data_list):
                                dataset.append(data_list[i])

                output_dict[idx] = dataset

        # 输出表格
        if output_dict:
            table = PrettyTable()
            table.field_names = ["用户ID", "类别数", "类别详情 (类别ID(样本数))", "总样本数"]
            table.align = "l"
            for idx in sorted(output_dict.keys()):
                classes = sorted(user_class_dict[idx]) if idx in user_class_dict else []
                # 构建类别详情字符串，格式：类别ID(样本数)
                class_details = []
                for cls in classes:
                    sample_count = user_class_samples[idx].get(cls, 0)
                    class_details.append(f"{cls}({sample_count})")
                classes_str = ", ".join(class_details)
                # 如果类别太多，截断显示
                if len(classes_str) > 100:
                    classes_str = classes_str[:97] + "..."
                
                total_samples = len(output_dict[idx])
                table.add_row([
                    f"User {idx}",
                    len(classes),
                    classes_str,
                    total_samples
                ])
            logger.info(f"\n{dataset_type}数据划分 (Baseline):\n{table}")

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

    @staticmethod
    def per_class_downsample(data_list, sample_ratio, rng=None):
        """
        按类别内样本比例采样（保持类别集合不变）。
        
        参数:
            data_list (list): Datum 对象列表。
            sample_ratio (float): 采样比例 (0.0 - 1.0)。
            rng (Random, optional): 随机数生成器。如果为 None，使用默认的 random 模块。
        
        返回:
            downsampled (list): 采样后的 Datum 对象列表。
        """
        if rng is None:
            rng = random
        
        by_class = defaultdict(list)
        for d in data_list:
            by_class[d.classname].append(d)
        
        downsampled = []
        for cname, items in by_class.items():
            n_keep = max(1, int(round(len(items) * sample_ratio)))
            if len(items) <= n_keep:
                downsampled.extend(items)
            else:
                downsampled.extend(rng.sample(items, n_keep))
        
        return downsampled

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """
        将训练验证集分割为训练集和验证集。
        
        参数:
            trainval (list): 训练验证集数据列表（Datum 对象列表）。
            p_val (float): 验证集比例，默认 0.2。
        
        返回:
            train (list): 训练集数据列表。
            val (list): 验证集数据列表。
        """
        logger = _get_dataset_logger()
        p_trn = 1 - p_val
        logger.info("Splitting trainval into %.0f%% train and %.0f%% val", p_trn * 100, p_val * 100)
        
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0, f"Class {label} has too few samples for validation split"
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
        """
        保存数据集分割到 JSON 文件。
        
        参数:
            train (list): 训练集数据列表（Datum 对象列表）。
            val (list): 验证集数据列表（Datum 对象列表）。
            test (list): 测试集数据列表（Datum 对象列表）。
            filepath (str): 保存路径。
            path_prefix (str): 路径前缀，用于相对化图像路径。
        """
        logger = _get_dataset_logger()
        
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
        """
        从 JSON 文件读取数据集分割。
        
        参数:
            filepath (str): 分割文件路径。
            path_prefix (str): 图像路径前缀。
        
        返回:
            train (list): 训练集数据列表（Datum 对象列表）。
            val (list): 验证集数据列表（Datum 对象列表）。
            test (list): 测试集数据列表（Datum 对象列表）。
        """
        logger = _get_dataset_logger()
        
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
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=None, new_cnames=None):
        """
        从按类别组织的目录结构中读取并分割数据。
        
        数据组织格式：
            images/
                dog/
                cat/
                horse/
        
        参数:
            image_dir (str): 图像目录路径。
            p_trn (float): 训练集比例，默认 0.5。
            p_val (float): 验证集比例，默认 0.2。
            ignored (list, optional): 要忽略的类别名称列表。
            new_cnames (dict, optional): 类别名称映射字典 {old_name: new_name}。
        
        返回:
            train (list): 训练集数据列表（Datum 对象列表）。
            val (list): 验证集数据列表（Datum 对象列表）。
            test (list): 测试集数据列表（Datum 对象列表）。
        """
        if ignored is None:
            ignored = []
        
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        logger = _get_dataset_logger()
        logger.info("Splitting into %.0f%% train, %.0f%% val, and %.0f%% test", 
                   p_trn * 100, p_val * 100, p_tst * 100)

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test

    def prepare_federated_data(self, train, test, cfg, train_sample_ratio=None, test_sample_ratio=None):
        """
        准备联邦学习数据的通用方法。
        
        参数:
            train (list): 训练集数据列表（Datum 对象列表）。
            test (list): 测试集数据列表（Datum 对象列表）。
            cfg: 配置对象，需要包含 DATASET.USEALL, DATASET.NUM_SHOTS, DATASET.USERS, DATASET.IID 等属性。
            train_sample_ratio (float, optional): 训练集采样比例。如果为 None，不进行采样。
            test_sample_ratio (float, optional): 测试集采样比例。如果为 None，不进行采样。
        
        返回:
            federated_train_x (dict): 联邦训练集 {user_id: [Datum_list]}。
            federated_test_x (dict): 联邦测试集 {user_id: [Datum_list]}。
        """
        # 按类别内样本比例采样（保持类别集合不变）
        rng = random.Random(getattr(cfg, 'SEED', 1))
        
        if train_sample_ratio is not None:
            train = self.per_class_downsample(train, train_sample_ratio, rng)
        
        if test_sample_ratio is not None:
            test = self.per_class_downsample(test, test_sample_ratio, rng)

        # 根据用户数量决定是否使用类别重叠
        # if cfg.DATASET.USERS >= 20:
        #     repeat_rate = 0.1
        # else:
        #     repeat_rate = 0

        repeat_rate = 0.0

        # 生成联邦数据集
        if cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(
                train,
                num_shots=cfg.DATASET.NUM_SHOTS,
                num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID,
                repeat_rate=repeat_rate,
                dataset_type="训练"
            )
        else:
            federated_train_x = self.generate_federated_fewshot_dataset(
                train,
                num_shots=cfg.DATASET.NUM_SHOTS,
                num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID,
                repeat_rate=repeat_rate,
                dataset_type="训练"
            )

        federated_test_x = self.generate_federated_dataset(
            test,
            num_shots=cfg.DATASET.NUM_SHOTS,
            num_users=cfg.DATASET.USERS,
            is_iid=cfg.DATASET.IID,
            repeat_rate=repeat_rate,
            dataset_type="测试"
        )

        return federated_train_x, federated_test_x