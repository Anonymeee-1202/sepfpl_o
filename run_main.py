import os
from datasets import download_standard_datasets

def run(root, dataset, users, factorization, rank, noise, seed):
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    os.system(f'bash srun_main.sh {root} {dataset_yaml} {users} {factorization} {rank} {noise} {seed}')

# variables
# seed_list = [1, 2, 3]
# seed_list = [1]
# dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers']
# factorization_list = ['dpfpl', 'fedpgp', 'dplora', 'promptfl', 'fedotp']  # 可选择的方法
# rank_list = [1, 2, 4, 8]
# noise_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]

root = '/home/liuxin25/dataset' # change to your dataset path
users = 10


def test_generalization_and_personalization():
    seed_list = [1]
    dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers']
    factorization_list = ['dpfpl', 'fedpgp', 'promptfl', 'fedotp']  # 可选择的方法
    noise_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
    for seed in seed_list:
        for dataset in dataset_list:
            for factorization in factorization_list:
                    for noise in noise_list:
                        run(root, dataset, users, factorization, 8, noise, seed)

def test_1():
    run(root, 'caltech101', users, 'dpfpl', 8, 0.0, 1)

def download_datasets(target_root):
    download_standard_datasets(target_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DP-FPL experiments")
    parser.add_argument("--test_generalization_and_personalization", action="store_true", help="运行个性化与泛化性测试批处理")
    parser.add_argument("--test1", action="store_true", help="运行单个测试: Caltech101 + DP-FPL + rank=8 + noise=0.0 + seed=1")
    parser.add_argument("--download", action="store_true", help="下载 Caltech101、OxfordPets、OxfordFlowers 到 root 目录")
    args = parser.parse_args()

    if args.download:
        download_datasets(root)
    elif args.test_generalization_and_personalization:
        test_generalization_and_personalization()
    elif args.test1:
        test_1()
    else:
        print("未指定操作。使用 --download 下载数据集，--test 运行测试批处理，或 --test1 运行单个测试。")
