import os
import json
import tarfile
import urllib.request
from urllib.error import URLError
from typing import List, Optional

# --- 数据常量 ---
# Oxford Flowers 102 的类别名称列表，索引对应类别 ID (1-102)
FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


def download_standard_datasets(base_root: str, dataset_list: Optional[List[str]] = None) -> None:
    """
    下载并标准化常用的计算机视觉数据集。

    功能说明:
        1. 调用 torchvision 下载指定数据集。
        2. 解决 torchvision 下载文件夹命名不统一的问题，将其重命名为标准格式。
        3. 对特定数据集 (如 Flowers102) 进行后处理，补充缺失的元数据文件。

    支持的数据集 keys:
        - caltech-101
        - oxford_pets
        - oxford_flowers
        - food-101
        - cifar-100
        - stanford_dogs

    参数:
        base_root (str): 数据集存储的根目录路径。会自动创建该目录。
        dataset_list (list, optional): 需要下载的数据集列表。
                                       如果为 None，默认下载 ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101']。

    Raises:
        ImportError: 如果未安装 torchvision。
    """
    os.makedirs(base_root, exist_ok=True)

    # --- 依赖检查 ---
    try:
        from torchvision import datasets as tvd
    except ImportError as e:
        print("❌ 错误: 未检测到 torchvision。请执行: pip install torchvision")
        raise e

    # --- 默认配置 ---
    if dataset_list is None:
        dataset_list = ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101']

    print(f"📂 数据集根目录: {base_root}")
    print(f"📋 任务列表: {', '.join(dataset_list)}")

    # --- 注册表配置 ---
    # 映射关系: 标准 key -> {显示名, torchvision类, 潜在的原始文件夹名}
    # candidates 用于捕获 torchvision 版本更新可能导致的文件夹命名变化
    registry = {
        'caltech-101': {
            'name': 'Caltech101',
            'downloader': tvd.Caltech101,
            'candidates': ['caltech101', 'Caltech101', 'caltech-101'],
        },
        'oxford_pets': {
            'name': 'Oxford-IIIT Pet',
            'downloader': tvd.OxfordIIITPet,
            'candidates': ['oxford-iiit-pet', 'OxfordIIITPet', 'oxford_pets'],
        },
        'oxford_flowers': {
            'name': 'Flowers102',
            'downloader': tvd.Flowers102,
            'candidates': ['flowers-102', 'Flowers102', 'oxford_flowers'],
        },
        'food-101': {
            'name': 'Food-101',
            'downloader': tvd.Food101,
            'candidates': ['food-101', 'Food101', 'food101'],
        },
        'cifar-100': {
            'name': 'CIFAR-100',
            'downloader': tvd.CIFAR100,
            'candidates': ['cifar-100', 'CIFAR100', 'cifar100'],
        },
        'stanford_dogs': {
            'name': 'Stanford Dogs',
            'downloader': None,  # 需要手动下载
            'candidates': ['stanford_dogs', 'StanfordDogs', 'stanford-dogs'],
        },
    }

    # --- 内部辅助函数 ---

    def ensure_standard_dir(standard_key: str) -> bool:
        """
        检查数据集目录是否存在。
        如果发现了 torchvision生成的非标准目录名 (candidates)，则将其重命名为标准 key。
        返回: True 表示目录已就绪 (无需下载)，False 表示需要下载。
        """
        target_dir = os.path.join(base_root, standard_key)
        
        # 1. 检查标准目录是否已存在
        if os.path.isdir(target_dir):
            # 对于 Stanford Dogs，还需要检查必要的子目录
            if standard_key == 'stanford_dogs':
                images_dir = os.path.join(target_dir, 'Images')
                annotation_dir = os.path.join(target_dir, 'Annotation')
                if os.path.isdir(images_dir) and os.path.isdir(annotation_dir):
                    return True
                else:
                    # 目录存在但结构不完整，需要重新下载
                    return False
            return True
            
        # 2. 检查是否存在别名目录 (由 torchvision 自动生成)，若有则重命名
        for cand in registry[standard_key]['candidates']:
            cand_path = os.path.join(base_root, cand)
            if os.path.isdir(cand_path):
                try:
                    print(f"🔄 检测到原始目录，正在标准化命名: {cand} -> {standard_key}")
                    os.rename(cand_path, target_dir)
                    return True
                except OSError as re:
                    print(f"⚠️ 重命名失败 ({cand} -> {standard_key}): {re}")
                    break # 停止尝试其他 candidate
        return False

    def download_stanford_dogs(download_root: str) -> None:
        """
        下载 Stanford Dogs 数据集。
        
        Stanford Dogs 数据集包含 120 个狗品种类别，需要从 Stanford 网站下载。
        数据集结构：
            stanford_dogs/
                Images/
                Annotation/
                file_list.mat
                test_list.mat
                train_list.mat
        
        参数:
            download_root (str): 数据集存储的根目录。
        """
        target_dir = os.path.join(download_root, 'stanford_dogs')
        os.makedirs(target_dir, exist_ok=True)
        
        # Stanford Dogs 数据集的下载链接
        urls = {
            'images': 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
            'annotations': 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar',
            'lists': 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar',
        }
        
        print("⬇️ 正在下载 Stanford Dogs 数据集...")
        print("   注意: 数据集较大 (~800MB)，下载可能需要一些时间。")
        
        for key, url in urls.items():
            filename = f'{key}.tar'
            filepath = os.path.join(target_dir, filename)
            extract_dir = target_dir
            
            # 检查是否需要下载
            need_download = not os.path.exists(filepath)
            
            # 检查是否需要解压（检查解压后的目录是否存在）
            need_extract = True
            if key == 'images':
                # 检查 Images 目录是否存在
                if os.path.isdir(os.path.join(target_dir, 'Images')):
                    need_extract = False
            elif key == 'annotations':
                # 检查 Annotation 目录是否存在
                if os.path.isdir(os.path.join(target_dir, 'Annotation')):
                    need_extract = False
            elif key == 'lists':
                # 检查列表文件是否存在
                list_files = ['file_list.mat', 'test_list.mat', 'train_list.mat']
                if all(os.path.exists(os.path.join(target_dir, f)) for f in list_files):
                    need_extract = False
            
            if not need_download and not need_extract:
                print(f"   ✓ {filename} 已存在且已解压，跳过")
                continue
            
            # 下载文件
            if need_download:
                try:
                    print(f"   ⬇️ 正在下载 {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                    print(f"   ✓ {filename} 下载完成")
                except Exception as e:
                    print(f"   ❌ {filename} 下载失败: {e}")
                    continue
            
            # 解压文件
            if need_extract and os.path.exists(filepath):
                try:
                    print(f"   📦 正在解压 {filename}...")
                    with tarfile.open(filepath, 'r') as tar:
                        tar.extractall(path=extract_dir)
                    print(f"   ✓ {filename} 解压完成")
                except Exception as e:
                    print(f"   ⚠️ {filename} 解压失败: {e}")
        
        # 验证目录结构
        images_dir = os.path.join(target_dir, 'Images')
        annotation_dir = os.path.join(target_dir, 'Annotation')
        
        if os.path.isdir(images_dir) and os.path.isdir(annotation_dir):
            print(f"✅ Stanford Dogs 数据集准备完成: {target_dir}")
        else:
            print(f"⚠️ 警告: 数据集目录结构可能不完整，请检查 {target_dir}")

    def post_setup(standard_key: str) -> None:
        """
        下载完成后的特定数据集处理逻辑。
        例如：生成标签映射文件，方便后续 DataLoader 使用。
        """
        if standard_key == 'oxford_flowers':
            ds_dir = os.path.join(base_root, standard_key)
            cat_file = os.path.join(ds_dir, 'cat_to_name.json')
            
            # 如果映射文件不存在，则创建
            if not os.path.isfile(cat_file):
                try:
                    # 索引从 1 开始，匹配 Flowers102 的文件夹命名习惯
                    mapping = {str(i + 1): name for i, name in enumerate(FLOWERS102_CLASSES)}
                    with open(cat_file, 'w') as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)
                    print(f"✨ 已生成类别映射文件: {cat_file}")
                except Exception as e:
                    print(f"⚠️ 生成 cat_to_name.json 失败: {e}")

    # --- 主下载循环 ---
    
    for key in dataset_list:
        standard_key = key.lower().strip()
        
        # 1. 校验 Key
        if standard_key not in registry:
            print(f"⚠️ 跳过未知数据集 Key: {key}")
            continue

        # 2. 检查是否已存在 (Pre-download check)
        # 如果本地已经有文件夹，直接跳过下载，并执行后处理检查
        if ensure_standard_dir(standard_key):
            print(f"✅ {registry[standard_key]['name']} 已存在，跳过下载。")
            post_setup(standard_key)
            continue

        # 3. 执行下载
        meta = registry[standard_key]
        print(f"⬇️ 正在下载 {meta['name']} ...")
        
        try:
            # Stanford Dogs 需要特殊处理
            if standard_key == 'stanford_dogs':
                download_stanford_dogs(base_root)
                print(f"🎉 {meta['name']} 下载完成。")
            else:
                # download=True 会触发 torchvision 的下载逻辑
                download_root = base_root
                if standard_key == 'cifar-100':
                    download_root = os.path.join(base_root, standard_key)
                    os.makedirs(download_root, exist_ok=True)
                if meta['downloader'] is not None:
                    meta['downloader'](root=download_root, download=True)
                    print(f"🎉 {meta['name']} 下载完成。")
                else:
                    print(f"⚠️ {meta['name']} 没有配置下载器，跳过下载。")
        except (URLError, Exception) as e:
            print(f"❌ {meta['name']} 下载失败: {e}")
            continue

        # 4. 下载后再次检查与标准化 (Post-download standardization)
        # torchvision 下载完后可能会生成默认命名的文件夹，需要再次运行重命名逻辑
        if ensure_standard_dir(standard_key):
            post_setup(standard_key)
        else:
            print(f"⚠️虽然下载未报错，但未找到预期目录。请检查 {base_root} 下的文件夹名称。")