#!/bin/bash

# MIA 实验任务列表: exp3-mia
# 生成时间: 2025-11-30 16:36:46
# 任务总数: 18
# 每个任务包含 2 个步骤：生成shadow数据（包含训练shadow模型） -> 训练攻击模型（训练完成后自动测试）
# 执行策略: 不同 GPU 的任务并行执行；同一 GPU 的任务串行执行。
# --------------------------------------------------------------------

# 并行执行模式 (多 GPU)

run_gpu_0() {
    echo "[Worker 0] 启动"
    # [1/18] caltech-101 | sepfpl | r=8 n=0.0 u=10 s=1
    echo '  --> [[1/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 0 1 5 exp3-mia '[1/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[1/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[1/18]] 生成Shadow数据 完成'
    echo '  --> [[1/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.0 1 5 exp3-mia '[1/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[1/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[1/18]] 训练攻击模型（含测试） 完成'
    # [3/18] caltech-101 | sepfpl | r=8 n=0.2 u=10 s=1
    echo '  --> [[3/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 0 1 5 exp3-mia '[3/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[3/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[3/18]] 生成Shadow数据 完成'
    echo '  --> [[3/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.2 1 5 exp3-mia '[3/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[3/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[3/18]] 训练攻击模型（含测试） 完成'
    # [5/18] caltech-101 | sepfpl | r=8 n=0.05 u=10 s=1
    echo '  --> [[5/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 0 1 5 exp3-mia '[5/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[5/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[5/18]] 生成Shadow数据 完成'
    echo '  --> [[5/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.05 1 5 exp3-mia '[5/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[5/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[5/18]] 训练攻击模型（含测试） 完成'
    # [7/18] oxford_pets | sepfpl | r=8 n=0.0 u=10 s=1
    echo '  --> [[7/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 0 1 5 exp3-mia '[7/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[7/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[7/18]] 生成Shadow数据 完成'
    echo '  --> [[7/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.0 1 5 exp3-mia '[7/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[7/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[7/18]] 训练攻击模型（含测试） 完成'
    # [9/18] oxford_pets | sepfpl | r=8 n=0.2 u=10 s=1
    echo '  --> [[9/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 0 1 5 exp3-mia '[9/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[9/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[9/18]] 生成Shadow数据 完成'
    echo '  --> [[9/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.2 1 5 exp3-mia '[9/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[9/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[9/18]] 训练攻击模型（含测试） 完成'
    # [11/18] oxford_pets | sepfpl | r=8 n=0.05 u=10 s=1
    echo '  --> [[11/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 0 1 5 exp3-mia '[11/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[11/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[11/18]] 生成Shadow数据 完成'
    echo '  --> [[11/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.05 1 5 exp3-mia '[11/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[11/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[11/18]] 训练攻击模型（含测试） 完成'
    # [13/18] oxford_flowers | sepfpl | r=8 n=0.0 u=10 s=1
    echo '  --> [[13/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 0 1 5 exp3-mia '[13/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[13/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[13/18]] 生成Shadow数据 完成'
    echo '  --> [[13/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.0 1 5 exp3-mia '[13/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[13/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[13/18]] 训练攻击模型（含测试） 完成'
    # [15/18] oxford_flowers | sepfpl | r=8 n=0.2 u=10 s=1
    echo '  --> [[15/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 0 1 5 exp3-mia '[15/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[15/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[15/18]] 生成Shadow数据 完成'
    echo '  --> [[15/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.2 1 5 exp3-mia '[15/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[15/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[15/18]] 训练攻击模型（含测试） 完成'
    # [17/18] oxford_flowers | sepfpl | r=8 n=0.05 u=10 s=1
    echo '  --> [[17/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=0 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 0 1 5 exp3-mia '[17/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[17/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[17/18]] 生成Shadow数据 完成'
    echo '  --> [[17/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=0 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.05 1 5 exp3-mia '[17/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[17/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[17/18]] 训练攻击模型（含测试） 完成'
    echo "[Worker 0] 完成"
}

run_gpu_1() {
    echo "[Worker 1] 启动"
    # [2/18] caltech-101 | sepfpl | r=8 n=0.4 u=10 s=1
    echo '  --> [[2/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 0 1 5 exp3-mia '[2/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[2/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[2/18]] 生成Shadow数据 完成'
    echo '  --> [[2/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.4 1 5 exp3-mia '[2/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[2/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[2/18]] 训练攻击模型（含测试） 完成'
    # [4/18] caltech-101 | sepfpl | r=8 n=0.1 u=10 s=1
    echo '  --> [[4/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 0 1 5 exp3-mia '[4/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[4/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[4/18]] 生成Shadow数据 完成'
    echo '  --> [[4/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.1 1 5 exp3-mia '[4/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[4/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[4/18]] 训练攻击模型（含测试） 完成'
    # [6/18] caltech-101 | sepfpl | r=8 n=0.01 u=10 s=1
    echo '  --> [[6/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 0 1 5 exp3-mia '[6/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[6/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[6/18]] 生成Shadow数据 完成'
    echo '  --> [[6/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/caltech-101.yaml 10 sepfpl 8 0.01 1 5 exp3-mia '[6/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[6/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[6/18]] 训练攻击模型（含测试） 完成'
    # [8/18] oxford_pets | sepfpl | r=8 n=0.4 u=10 s=1
    echo '  --> [[8/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 0 1 5 exp3-mia '[8/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[8/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[8/18]] 生成Shadow数据 完成'
    echo '  --> [[8/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.4 1 5 exp3-mia '[8/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[8/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[8/18]] 训练攻击模型（含测试） 完成'
    # [10/18] oxford_pets | sepfpl | r=8 n=0.1 u=10 s=1
    echo '  --> [[10/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 0 1 5 exp3-mia '[10/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[10/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[10/18]] 生成Shadow数据 完成'
    echo '  --> [[10/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.1 1 5 exp3-mia '[10/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[10/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[10/18]] 训练攻击模型（含测试） 完成'
    # [12/18] oxford_pets | sepfpl | r=8 n=0.01 u=10 s=1
    echo '  --> [[12/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 0 1 5 exp3-mia '[12/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[12/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[12/18]] 生成Shadow数据 完成'
    echo '  --> [[12/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_pets.yaml 10 sepfpl 8 0.01 1 5 exp3-mia '[12/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[12/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[12/18]] 训练攻击模型（含测试） 完成'
    # [14/18] oxford_flowers | sepfpl | r=8 n=0.4 u=10 s=1
    echo '  --> [[14/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 0 1 5 exp3-mia '[14/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[14/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[14/18]] 生成Shadow数据 完成'
    echo '  --> [[14/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.4 1 5 exp3-mia '[14/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[14/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[14/18]] 训练攻击模型（含测试） 完成'
    # [16/18] oxford_flowers | sepfpl | r=8 n=0.1 u=10 s=1
    echo '  --> [[16/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 0 1 5 exp3-mia '[16/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[16/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[16/18]] 生成Shadow数据 完成'
    echo '  --> [[16/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.1 1 5 exp3-mia '[16/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[16/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[16/18]] 训练攻击模型（含测试） 完成'
    # [18/18] oxford_flowers | sepfpl | r=8 n=0.01 u=10 s=1
    echo '  --> [[18/18]] 生成Shadow数据'
    CUDA_VISIBLE_DEVICES=1 bash srun_generate_shadow.sh /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 0 1 5 exp3-mia '[18/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[18/18]] 生成Shadow数据 失败'
      return 1
    fi
    echo '  ✅ [[18/18]] 生成Shadow数据 完成'
    echo '  --> [[18/18]] 训练攻击模型（含测试）'
    CUDA_VISIBLE_DEVICES=1 bash srun_mia.sh train /home/liuxin25/dataset configs/datasets/oxford_flowers.yaml 10 sepfpl 8 0.01 1 5 exp3-mia '[18/18]'
    if [ $? -ne 0 ]; then
      echo '❌ [[18/18]] 训练攻击模型（含测试） 失败'
      return 1
    fi
    echo '  ✅ [[18/18]] 训练攻击模型（含测试） 完成'
    echo "[Worker 1] 完成"
}

echo '🚀 启动后台并行任务...'
run_gpu_0 &
run_gpu_1 &

wait
echo '✅ 所有任务已执行完毕。'
