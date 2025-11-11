from prettytable import PrettyTable
import pickle
import os
from statistics import mean, stdev


DEFAULT_DATASET_LIST = ['caltech-101', 'oxford_pets', 'oxford_flowers', 'food-101']
DEFAULT_FACTORIZATION_LIST = ['sepfpl', 'dpfpl', 'fedpgp', 'fedotp', 'promptfl']
DEFAULT_NOISE_LIST = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
DEFAULT_SEED_LIST = [1]
DEFAULT_RANK_LIST = [8]
TAIL_EPOCHS = 3


def load_config_from_run_main():
    try:
        from run_main import EXPERIMENT_CONFIG

        dataset_list = EXPERIMENT_CONFIG.get('dataset_list', DEFAULT_DATASET_LIST)
        factorization_list = EXPERIMENT_CONFIG.get('factorization_list', DEFAULT_FACTORIZATION_LIST)
        noise_list = EXPERIMENT_CONFIG.get('noise_list', DEFAULT_NOISE_LIST)
        seed_list = EXPERIMENT_CONFIG.get('seed_list', DEFAULT_SEED_LIST)
        rank_value = EXPERIMENT_CONFIG.get('rank', DEFAULT_RANK_LIST[0])
        rank_list = [rank_value] if not isinstance(rank_value, (list, tuple)) else list(rank_value)
        return dataset_list, factorization_list, noise_list, seed_list, rank_list
    except Exception:
        return (
            DEFAULT_DATASET_LIST,
            DEFAULT_FACTORIZATION_LIST,
            DEFAULT_NOISE_LIST,
            DEFAULT_SEED_LIST,
            DEFAULT_RANK_LIST,
        )


dataset_list, factorization_list, noise_list, seed_list, rank_list = load_config_from_run_main()


def tail_values(values, tail=TAIL_EPOCHS):
    if not values:
        return []
    if tail is None or len(values) <= tail:
        return list(values)
    return list(values[-tail:])


def load_metrics(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return [], []

    if isinstance(data, (list, tuple)):
        if len(data) >= 2 and isinstance(data[0], list):
            local_hist = data[0]
            neighbor_hist = data[1] if len(data) > 1 else []
        elif len(data) >= 6:
            local_hist, neighbor_hist = data[0], data[1]
        else:
            local_hist, neighbor_hist = [], []
    elif isinstance(data, dict):
        local_hist = data.get('local_acc', [])
        neighbor_hist = data.get('neighbor_acc', [])
    else:
        local_hist, neighbor_hist = [], []

    return local_hist or [], neighbor_hist or []


def format_stats(values):
    if not values:
        return '0.000 ± 0.000'
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.2f} ± {std:.2f}'


def read_data(dataset, factorization, rank, noise):
    per_seed_local, per_seed_neighbor = [], []
    for seed in seed_list:
        file_name = os.path.join(
            os.getcwd(), f'outputs/{dataset}/acc_{factorization}_{rank}_{noise}_{seed}.pkl'
        )
        if not os.path.isfile(file_name):
            continue

        local_hist, neighbor_hist = load_metrics(file_name)
        if local_hist:
            per_seed_local.extend(tail_values(local_hist))
        if neighbor_hist:
            per_seed_neighbor.extend(tail_values(neighbor_hist))

    local_stats = format_stats(per_seed_local)
    neighbor_stats = format_stats(per_seed_neighbor)
    return local_stats, neighbor_stats


def read_scheme(dataset, rank, noise):
    local_list, neighbor_list = [], []
    for factorization in factorization_list:
        local, neighbor = read_data(dataset, factorization, rank, noise)
        local_list.append(local)
        neighbor_list.append(neighbor)
    return local_list, neighbor_list


for dataset in dataset_list:
    headers = ['rank', 'noise'] + factorization_list
    local_table = PrettyTable(headers)
    neighbor_table = PrettyTable(headers)

    for rank in rank_list:
        for noise in noise_list:
            local_list, neighbor_list = read_scheme(dataset, rank, noise)
            local_row = [rank, noise] + local_list
            neighbor_row = [rank, noise] + neighbor_list
            local_table.add_row(local_row)
            neighbor_table.add_row(neighbor_row)

    print(f'========== {dataset} local accuracy ==========')
    print(local_table)
    print(f'========== {dataset} neighbor accuracy ==========')
    print(neighbor_table)
    print('\n\n')

