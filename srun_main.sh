#!/bin/bash

# 参数说明：
# $1: root - 数据集路径
# $2: dataset-config-file - 数据集配置文件路径
# $3: num-users - 客户端数量
# $4: factorization - 分解方法
# $5: rank - 矩阵分解的秩
# $6: noise - 差分隐私噪声级别
# $7: seed - 随机种子
# $8: round - 训练轮次
# $9: wandb-group - wandb group 参数（可选，留空则不传递）
# $10: task-id - 任务编号标识（可选，留空则不传递）
# $11+: 额外参数

WAND_GROUP="${9:-}"
TASK_ID="${10:-}"

PY_ARGS=(
  --root "$1"
  --dataset-config-file "$2"
  --num-users "$3"
  --factorization "$4"
  --rank "$5"
  --noise "$6"
  --seed "$7"
  --round "$8"
)

if [ -n "$WAND_GROUP" ]; then
  PY_ARGS+=(--wandb-group "$WAND_GROUP")
fi

if [ -n "$TASK_ID" ]; then
  PY_ARGS+=(--task-id "$TASK_ID")
fi

if [ "$#" -ge 11 ]; then
  for extra_arg in "${@:11}"; do
    PY_ARGS+=("$extra_arg")
  done
fi

python federated_main.py "${PY_ARGS[@]}"
