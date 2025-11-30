#!/bin/bash

# Shadow 数据生成脚本
# 此脚本用于生成 MIA 攻击训练所需的 shadow 数据
# 对于每个 seed，先训练 shadow 模型，然后生成 shadow 数据
# 参数说明：
# $1: root - 数据集路径
# $2: dataset-config-file - 数据集配置文件路径
# $3: num-users - 客户端数量
# $4: factorization - 分解方法
# $5: rank - 矩阵分解的秩
# $6: noise - 差分隐私噪声级别
# $7: start-seed - 起始随机种子
# $8: end-seed - 结束随机种子（包含）
# $9: round - 训练轮次
# $10: wandb-group - wandb group 参数（可选，留空则不传递）
# $11: task-id - 任务编号标识（可选，留空则不传递）

WAND_GROUP="${10:-}"
TASK_ID="${11:-}"

START_SEED=$7
END_SEED=$8

echo "=========================================="
echo "生成 Shadow 数据"
echo "数据集: $2"
echo "Seed 范围: $START_SEED - $END_SEED"
echo "=========================================="

for seed in $(seq $START_SEED $END_SEED); do
  echo ""
  echo "----------------------------------------"
  echo "正在处理 Shadow 模型 (seed=$seed)"
  echo "----------------------------------------"
  
  # 训练 shadow 模型并生成 shadow 数据（集成在一个命令中）
  echo "  训练 shadow 模型并生成 shadow 数据..."
  PY_ARGS=(
    --root "$1"
    --dataset-config-file "$2"
    --num-users "$3"
    --factorization "$4"
    --rank "$5"
    --noise "$6"
    --seed "$seed"
    --round "$9"
    --skip-test
    --generate-shadow
  )

  if [ -n "$WAND_GROUP" ]; then
    PY_ARGS+=(--wandb-group "$WAND_GROUP")
  fi

  if [ -n "$TASK_ID" ]; then
    PY_ARGS+=(--task-id "$TASK_ID")
  fi

  python federated_main.py "${PY_ARGS[@]}"
  
  if [ $? -ne 0 ]; then
    echo "❌ Shadow 模型训练或数据生成失败 (seed=$seed)"
    exit 1
  fi
  
  echo "✅ Shadow 数据生成完成 (seed=$seed)"
done

echo ""
echo "=========================================="
echo "所有 Shadow 数据生成完成"
echo "=========================================="

