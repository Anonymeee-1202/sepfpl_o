import os
from urllib.error import URLError
import json

FLOWERS102_CLASSES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def download_standard_datasets(base_root: str, dataset_list: list = None) -> None:
    """
    根据 dataset_list 下载数据集到 base_root：
    - caltech-101 (Caltech-101)
    - oxford_pets (Oxford-IIIT Pet)
    - oxford_flowers (Oxford Flowers 102)
    - food-101 (Food-101)

    参数:
        base_root: 数据集保存的根目录
        dataset_list: 要下载的数据集列表，默认为 None（下载所有默认数据集）

    依赖 torchvision。
    """
    os.makedirs(base_root, exist_ok=True)
    try:
        from torchvision import datasets as tvd
    except Exception as e:
        print("未安装 torchvision，请先执行: pip install torchvision")
        raise e

    # 如果未指定，使用默认数据集列表
    if dataset_list is None:
        dataset_list = ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101']

    print(f"下载目标目录: {base_root}")
    print(f"将下载数据集: {', '.join(dataset_list)}")

    # 统一配置：显示名、下载器、可能的原始目录名（用于重命名）
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
    }

    def ensure_standard_dir(standard_key: str) -> bool:
        """若标准目录已存在或可通过候选名重命名得到，返回 True（表示无需下载）。"""
        target_dir = os.path.join(base_root, standard_key)
        if os.path.isdir(target_dir):
            return True
        for cand in registry[standard_key]['candidates']:
            cand_path = os.path.join(base_root, cand)
            if os.path.isdir(cand_path):
                try:
                    print(f"检测到已下载的目录，重命名 {cand_path} -> {target_dir}")
                    os.rename(cand_path, target_dir)
                    return True
                except Exception as re:
                    print(f"⚠️  预重命名失败: {re}")
                    break
        return False

    def post_setup(standard_key: str) -> None:
        """下载后数据集特定的补全逻辑。"""
        if standard_key == 'oxford_flowers':
            ds_dir = os.path.join(base_root, standard_key)
            # 生成 cat_to_name.json（若不存在）
            cat_file = os.path.join(ds_dir, 'cat_to_name.json')
            if not os.path.isfile(cat_file):
                try:
                    mapping = {str(i + 1): name for i, name in enumerate(FLOWERS102_CLASSES)}
                    with open(cat_file, 'w') as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)
                    print(f"已生成 {cat_file} (Oxford Flowers 102 类别映射)")
                except Exception as e:
                    print(f"⚠️  生成 cat_to_name.json 失败: {e}")

    for key in dataset_list:
        standard_key = key.lower().strip()
        if standard_key not in registry:
            print(f"⚠️  未知数据集: {key}，跳过")
            continue

        # 若已存在（或可预重命名得到），跳过下载
        if ensure_standard_dir(standard_key):
            print(f"✅ 检测到已存在的数据集目录，跳过下载: {os.path.join(base_root, standard_key)}")
            post_setup(standard_key)
            continue

        name = registry[standard_key]['name']
        downloader = registry[standard_key]['downloader']

        print(f"下载 {name} 到 {base_root} ...")
        try:
            downloader(root=base_root, download=True)
            print(f"{name} 下载完成")
        except (URLError, Exception) as e:
            print(f"❌ {name} 下载失败: {e}")
            continue

        # 下载后再次尝试标准化目录名
        if ensure_standard_dir(standard_key):
            post_setup(standard_key)
            continue

        # 若仍未找到可重命名的目录，提示手动检查
        print(f"⚠️  未找到可重命名到标准目录的源目录，请检查: {name}")


