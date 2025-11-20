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
# $9: wandb-group - wandb group 参数（必选）
# $10: task-id - 任务编号标识（必选）
# $11+: 额外参数

python federated_main.py \
  --root "$1" \
  --dataset-config-file "$2" \
  --num-users "$3" \
  --factorization "$4" \
  --rank "$5" \
  --noise "$6" \
  --seed "$7" \
  --round "$8" \
  --wandb-group "$9" \
  --task-id "${10}" \
  "${@:11}"  # 传递第11个参数及之后的所有额外参数
