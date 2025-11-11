#!/bin/bash

# 第8个参数：数据划分策略（partition）
PARTITION=${8:-noniid-labeldir}
# 第9个参数：训练轮次
ROUND=${9:-10}
# 可选第10个参数：显卡列表，如 0 或 0,1
GPUS_ARG=${10:-}
if [ -n "$GPUS_ARG" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS_ARG"
fi

python federated_main.py \
  --root $1 \
  --dataset-config-file $2 \
  --num-users $3 \
  --factorization $4 \
  --rank $5 \
  --noise $6 \
  --seed $7 \
  --partition $PARTITION \
  --round $ROUND \
  ${GPUS_ARG:+--gpus $GPUS_ARG}
