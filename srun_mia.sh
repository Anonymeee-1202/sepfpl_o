#!/bin/bash

# MIA 攻击模型训练/测试/生成shadow数据脚本
# 训练模式会自动在训练完成后进行测试
# 参数说明：
# $1: mode - 模式：'train'、'test' 或 'generate_shadow'
# 
# 训练模式参数（训练完成后自动测试，需要测试相关参数）：
#   $2: root - 数据集路径
#   $3: dataset-config-file - 数据集配置文件路径
#   $4: num-users - 客户端数量
#   $5: factorization - 分解方法
#   $6: rank - 矩阵分解的秩
#   $7: noise - 差分隐私噪声级别
#   $8: seed - 随机种子
#   $9: round - 训练轮次（用于加载对应的checkpoint）
#   $10: wandb-group - wandb group 参数（可选，留空则不传递）
#   $11: task-id - 任务编号标识（可选，留空则不传递）
#   $12+: 额外参数（包括 --noise-list 和 --shadow-sample-ratio-list）
#
# 测试模式参数（仅测试，不训练）：
#   $2: root - 数据集路径
#   $3: dataset-config-file - 数据集配置文件路径
#   $4: num-users - 客户端数量
#   $5: factorization - 分解方法
#   $6: rank - 矩阵分解的秩
#   $7: noise - 差分隐私噪声级别
#   $8: seed - 随机种子
#   $9: round - 训练轮次（用于加载对应的checkpoint）
#   $10: wandb-group - wandb group 参数（可选，留空则不传递）
#   $11: task-id - 任务编号标识（可选，留空则不传递）
#   $12+: 额外参数（包括 --noise-list 和 --shadow-sample-ratio-list）
#
# 生成shadow数据模式参数（从checkpoint生成shadow数据）：
#   $2: root - 数据集路径
#   $3: dataset-config-file - 数据集配置文件路径
#   $4: num-users - 客户端数量
#   $5: factorization - 分解方法
#   $6: rank - 矩阵分解的秩
#   $7: noise - 差分隐私噪声级别
#   $8: seed - 随机种子（用于定位checkpoint）
#   $9: round - 训练轮次（用于加载对应的checkpoint）
#   $10: wandb-group - wandb group 参数（可选，留空则不传递）
#   $11: task-id - 任务编号标识（可选，留空则不传递）
#   $12+: 额外参数（包括 --noise-list 和 --shadow-sample-ratio-list）

MODE="$1"

if [ -z "$MODE" ]; then
    echo "错误: 必须指定模式 'train'、'test' 或 'generate_shadow'"
    echo "用法: $0 <train|test|generate_shadow> [参数...]"
    exit 1
fi

if [ "$MODE" != "train" ] && [ "$MODE" != "test" ] && [ "$MODE" != "generate_shadow" ]; then
    echo "错误: 模式必须是 'train'、'test' 或 'generate_shadow'"
    exit 1
fi

# 训练和测试模式使用相同的参数（因为训练后会自动测试）
WAND_GROUP="${10:-}"
TASK_ID="${11:-}"

PY_ARGS=(
    --mode "$MODE"
    --root "$2"
    --dataset-config-file "$3"
    --num-users "$4"
    --factorization "$5"
    --rank "$6"
    --noise "$7"
    --seed "$8"
    --round "$9"
)

if [ -n "$WAND_GROUP" ]; then
    PY_ARGS+=(--wandb-group "$WAND_GROUP")
fi

if [ -n "$TASK_ID" ]; then
    PY_ARGS+=(--task-id "$TASK_ID")
fi

if [ "$#" -ge 12 ]; then
    for extra_arg in "${@:12}"; do
        PY_ARGS+=("$extra_arg")
    done
fi

python mia.py "${PY_ARGS[@]}"

