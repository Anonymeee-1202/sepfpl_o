# SepFPL: Structural Entropy Enhanced Privacy Preserving Personalized Federated Prompt Learning

This repository contains the official implementation of the paper **"SepFPL: Structural Entropy Enhanced Privacy Preserving Personalized Federated Prompt Learning"**.

## Acknowledgments

This codebase is built on top of several open-source projects:

- **Dassl Toolbox**: The code structure and framework are based on the [Dassl](https://github.com/KaiyangZhou/CoOp/tree/main) toolbox from the CoOp project. We thank the authors for providing this excellent foundation.

- **Baseline Methods**: The implementations of baseline methods (DP-FPL, PromptFL, FedOTP, FedPGP) are adapted from [DP-FPL](https://github.com/linhhtran/DP-FPL). We acknowledge the original authors for their contributions.

## Installation

### Prerequisites

1. Install PyTorch following the [official instructions](https://pytorch.org/).

2. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

Place your datasets in the directory specified by `ROOT_DIR` (default: `~/dataset`). The code supports the following datasets:
- Caltech-101
- Oxford Flowers
- Food-101
- Stanford Dogs
- CIFAR-100

Dataset configuration files should be placed in `configs/datasets/`.

## Usage

### Quick Start

Run a single experiment:
```bash
python run_main.py --test \
    --dataset caltech-101 \
    --users 10 \
    --factorization sepfpl \
    --rank 8 \
    --noise 0.2 \
    --seed 1
```

### Batch Experiments

Generate batch experiment scripts:

**Experiment 1 (Standard):**
```bash
python run_main.py --exp1 --gpus 0,1 --threads 1
```

**Experiment 2 (MIA):**
```bash
python run_main.py --exp2 --gpus 0,1 --threads 1
```

This will generate bash scripts in the `scripts/` directory. Execute them to run the experiments:
```bash
bash scripts/run_exp1-standard.sh
```

### MIA Experiment Stages

For MIA experiments, you can control different stages:

```bash
python run_main.py --exp2 \
    --fed-train \
    --generate-shadow \
    --attack-train \
    --attack-test \
    --gpus 0,1
```

- `--fed-train`: Train the federated model
- `--generate-shadow`: Generate shadow model data
- `--attack-train`: Train the attack model
- `--attack-test`: Test the attack model

By default, `--generate-shadow` and `--attack-train` are enabled.

### Available Methods

The code supports the following factorization methods:
- `sepfpl`: Our proposed SepFPL method
- `dpfpl`: DP-FPL baseline
- `promptfl`: PromptFL baseline
- `fedotp`: FedOTP baseline
- `fedpgp`: FedPGP baseline

### Key Parameters

- `--gpus`: Comma-separated GPU IDs (e.g., `0,1`)
- `--threads`: Number of parallel tasks per GPU (default: 1)
- `--factorization`: Factorization method name
- `--rank`: Matrix rank for factorization
- `--noise`: Differential privacy noise level
- `--sepfpl-topk`: Top-k parameter for SepFPL (default: 8)
- `--rdp-p`: RDP time-adaptive power parameter (default: 0.2)

## Results and Visualization

### Generate Tables

Generate result tables:
```bash
python table.py --exp1 --output-dir ~/code/sepfpl/outputs
python table.py --exp2 --output-dir ~/code/sepfpl/outputs
```

### Generate Plots

Generate visualization plots:
```bash
python plot.py --mia-analysis --output-dir ~/code/sepfpl/outputs
```

## Project Structure

```
sepfpl_o/
├── federated_main.py      # Main federated learning training script
├── run_main.py           # Experiment management and script generation
├── table.py              # Result table generation
├── plot.py               # Visualization plotting
├── mia.py                # Membership inference attack implementation
├── srun_main.sh          # Training script wrapper
├── srun_mia.sh           # MIA script wrapper
├── configs/              # Configuration files
├── datasets/             # Dataset utilities
├── trainers/             # Training modules
└── utils/                # Utility functions
```

## License

This project is licensed under the MIT License.
