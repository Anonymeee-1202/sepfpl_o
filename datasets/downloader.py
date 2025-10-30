import os


def download_standard_datasets(target_root: str) -> None:
    """
    下载以下数据集到 target_root：
    - caltech101 (Caltech-101)
    - oxford_pets (Oxford-IIIT Pet)
    - oxford_flowers (Oxford Flowers 102)

    依赖 torchvision。
    """
    os.makedirs(target_root, exist_ok=True)
    try:
        from torchvision import datasets as tvd
    except Exception as e:
        print("未安装 torchvision，请先执行: pip install torchvision")
        raise e

    print(f"下载目标目录: {target_root}")

    # Caltech101
    try:
        print("下载 Caltech101 ...")
        tvd.Caltech101(root=target_root, download=True)
        print("Caltech101 下载完成")
    except Exception as e:
        print(f"Caltech101 下载失败: {e}")

    # Oxford-IIIT Pet
    try:
        print("下载 Oxford-IIIT Pet ...")
        tvd.OxfordIIITPet(root=target_root, download=True)
        print("Oxford-IIIT Pet 下载完成")
    except Exception as e:
        print(f"Oxford-IIIT Pet 下载失败: {e}")

    # Flowers102
    try:
        print("下载 Flowers102 ...")
        tvd.Flowers102(root=target_root, download=True)
        print("Flowers102 下载完成")
    except Exception as e:
        print(f"Flowers102 下载失败: {e}")


