import os
import time
from urllib.error import URLError


def download_standard_datasets(base_root: str, dataset_list) -> None:
    """
    根据 dataset_list 下载数据集到 base_root：
    - caltech101 (Caltech-101)
    - oxford_pets (Oxford-IIIT Pet)
    - oxford_flowers (Oxford Flowers 102)
    - food101 (Food-101)

    参数:
        base_root: 数据集保存的根目录
        dataset_list: 要下载的数据集列表，可以是字符串（单个数据集）或列表（多个数据集）
    
    依赖 torchvision。
    """
    os.makedirs(base_root, exist_ok=True)
    try:
        from torchvision import datasets as tvd
    except Exception as e:
        print("未安装 torchvision，请先执行: pip install torchvision")
        raise e

    # 标准化 dataset_list 为列表格式
    if isinstance(dataset_list, str):
        dataset_list = [dataset_list]
    elif dataset_list is None:
        # 如果未指定，下载所有默认数据集
        dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers', 'food101']

    print(f"下载目标目录: {base_root}")
    print(f"将下载数据集: {', '.join(dataset_list)}")

    # 数据集名称到下载函数的映射
    dataset_downloaders = {
        'caltech-101': ('Caltech101', tvd.Caltech101),
        'oxford_pets': ('Oxford-IIIT Pet', tvd.OxfordIIITPet),
        'oxford_flowers': ('Flowers102', tvd.Flowers102),
        'food-101': ('Food-101', tvd.Food101),
    }

    # 根据 dataset_list 下载指定的数据集
    for dataset_key in dataset_list:
        dataset_key_normalized = dataset_key.lower().strip()
        if dataset_key_normalized in dataset_downloaders:
            name, downloader = dataset_downloaders[dataset_key_normalized]
            # 为每个数据集创建单独的子目录：base_root/dataset
            dataset_dir = os.path.join(base_root, dataset_key_normalized)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 添加重试机制处理SSL错误
            max_retries = 3
            retry_delay = 5  # 秒
            success = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"第 {attempt} 次尝试下载 {name} ...")
                    else:
                        print(f"下载 {name} 到 {dataset_dir} ...")
                    downloader(root=dataset_dir, download=True)
                    print(f"{name} 下载完成")
                    success = True
                    break
                except (URLError, Exception) as e:
                    error_msg = str(e)
                    if "SSL" in error_msg or "EOF" in error_msg:
                        if attempt < max_retries:
                            print(f"⚠️  SSL/网络错误: {error_msg}")
                            print(f"等待 {retry_delay} 秒后重试 ({attempt}/{max_retries})...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                        else:
                            print(f"❌ {name} 下载失败（已重试 {max_retries} 次）: {error_msg}")
                            print(f"   提示：这通常是网络连接问题，可以稍后手动重试")
                    else:
                        print(f"❌ {name} 下载失败: {error_msg}")
                        break
            
            if not success:
                print(f"💡 建议：检查网络连接或稍后手动下载 {name}")
        else:
            print(f"⚠️  未知数据集: {dataset_key}，跳过")


