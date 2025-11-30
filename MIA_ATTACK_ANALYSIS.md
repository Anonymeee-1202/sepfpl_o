# MIA攻击成功率相同问题分析

## 问题现象

在 exp3-mia 实验中，每个数据集内不同 noise 值的 MIA 攻击成功率完全相同：

- **caltech-101**: 所有 noise 值都是 0.3941
- **oxford_flowers**: 所有 noise 值都是 0.4228  
- **oxford_pets**: 所有 noise 值都是 0.5547

## MIA攻击完整流程

### 1. 训练目标模型 (federated_main.py)

**流程**：
- 使用不同的 noise 值训练联邦学习模型
- 保存 checkpoint: `{factorization}_{rank}_{noise}_{seed}_{num_users}.pth.tar`

**问题**：
- 在 `federated_main.py` 中，`std` 变量在训练循环之前只计算一次
- 即使使用了时间自适应隐私分配（sepfpl），`local_trainer.std` 会在每个 epoch 更新
- 但实际使用的 `std` 变量仍然是初始值，导致所有 epoch 都使用相同的噪声标准差
- **结果**：所有不同 noise 值的模型参数完全相同

**证据**：
```bash
# 检查不同noise值的checkpoint参数
sepfpl_8_0.0_1_10.pth.tar: mean=0.000101, std=0.020197
sepfpl_8_0.01_1_10.pth.tar: mean=0.000101, std=0.020197
sepfpl_8_0.4_1_10.pth.tar: mean=0.000101, std=0.020197
```

### 2. 生成Shadow数据 (federated_main.py + generate_shadow_data)

**流程**：
- 使用不同的 noise 值训练 shadow 模型（seed 从 0 到 19）
- 收集每个 shadow 模型的预测结果
- 保存 shadow 数据: `shadow_{noise}_{seed}.pkl`

**问题**：
- Shadow 模型训练时也使用了相同的 bug（std 计算错误）
- 所有不同 noise 值的 shadow 模型完全相同
- **结果**：所有不同 noise 值的 shadow 数据完全相同

**证据**：
```bash
# 检查不同noise值的shadow数据
shadow_0.0_0.pkl: 前5个样本哈希=f46d1f3c
shadow_0.01_0.pkl: 前5个样本哈希=f46d1f3c
shadow_0.4_0.pkl: 前5个样本哈希=f46d1f3c
```

### 3. 训练攻击模型 (mia.py + train_attack_models)

**流程**：
- 对于每个 noise 值，加载对应的 shadow 数据（pattern: `shadow_{noise}_*.pkl`）
- 使用 shadow 数据训练攻击模型
- 保存攻击模型: `mia_{label}_{noise}.pth.tar`

**问题**：
- 由于所有 noise 值的 shadow 数据完全相同
- 训练出的攻击模型也完全相同
- **结果**：所有不同 noise 值的攻击模型相同

**证据**：
```bash
# 攻击模型文件大小几乎相同（只有微小差异可能是保存时间不同）
mia_10_0.0.pth.tar: size=8739 bytes
mia_10_0.01.pth.tar: size=8749 bytes
mia_10_0.4.pth.tar: size=8739 bytes
```

### 4. 测试攻击模型 (mia.py + test_attack_models)

**流程**：
- 对于每个 noise 值：
  1. 加载对应的目标模型（checkpoint）
  2. 加载对应的攻击模型
  3. 在目标模型上获取预测概率
  4. 使用攻击模型判断是否为成员
  5. 计算攻击成功率

**问题**：
- 由于目标模型相同，预测概率相同
- 由于攻击模型相同，判断结果相同
- **结果**：所有不同 noise 值的攻击成功率完全相同

## 根本原因

**核心问题**：`federated_main.py` 中 `std` 计算的位置错误

**原始代码（错误）**：
```python
# 在训练循环之前计算（只计算一次）
if args.noise > 0:
    std = local_trainer.std / cfg.DATASET.USERS

# 训练循环
for epoch in range(start_epoch, max_epoch):
    # 如果使用时间自适应，会更新 local_trainer.std
    if use_time_adaptive_flag:
        local_trainer.update_std_for_round(epoch)
    
    # 但这里使用的 std 仍然是初始值！
    if args.noise > 0:
        noise = torch.normal(0, std, ...)  # 使用了错误的 std
```

**修复后的代码**：
```python
# 训练循环
for epoch in range(start_epoch, max_epoch):
    # 先更新 local_trainer.std（如果使用时间自适应）
    if use_time_adaptive_flag:
        local_trainer.update_std_for_round(epoch)
    
    # 然后重新计算 std（使用更新后的 local_trainer.std）
    if args.noise > 0:
        std = local_trainer.std / cfg.DATASET.USERS
    
    # 现在使用正确的 std
    if args.noise > 0:
        noise = torch.normal(0, std, ...)  # 使用了正确的 std
```

## 影响范围

1. **目标模型训练**：所有 noise 值的模型相同
2. **Shadow 模型训练**：所有 noise 值的 shadow 模型相同
3. **Shadow 数据生成**：所有 noise 值的 shadow 数据相同
4. **攻击模型训练**：所有 noise 值的攻击模型相同
5. **攻击成功率**：所有 noise 值的攻击成功率相同

## 解决方案

1. **已修复**：`federated_main.py` 中 `std` 的计算位置
2. **需要重新训练**：
   - 重新训练所有目标模型（使用不同的 noise 值）
   - 重新生成所有 shadow 数据
   - 重新训练所有攻击模型
   - 重新测试攻击成功率

## 验证方法

修复后，可以通过以下方式验证：

1. **检查目标模型参数**：
   ```bash
   # 不同noise值的模型参数应该不同
   python -c "import torch; ..."
   ```

2. **检查Shadow数据**：
   ```bash
   # 不同noise值的shadow数据应该不同
   python -c "import pickle; ..."
   ```

3. **检查攻击成功率**：
   ```bash
   # 运行 generate_mia_table，应该看到不同的攻击成功率
   python table.py --mia-exp-name exp3-mia
   ```

## 总结

每个数据集内不同 noise 值的 MIA 攻击成功率相同，是因为：

1. **训练阶段**：由于 `std` 计算 bug，所有 noise 值的模型训练时使用了相同的噪声标准差
2. **模型结果**：所有 noise 值的模型参数完全相同
3. **Shadow数据**：所有 noise 值的 shadow 数据完全相同
4. **攻击模型**：所有 noise 值的攻击模型完全相同
5. **攻击结果**：所有 noise 值的攻击成功率完全相同

这是一个**级联错误**：训练阶段的 bug 导致后续所有环节都受到影响。

