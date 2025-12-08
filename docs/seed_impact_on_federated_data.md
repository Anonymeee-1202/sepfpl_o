# Seed 对联邦数据集生成的影响分析

## 问题发现

检查代码后发现，**seed 对联邦数据集生成有部分影响，但存在硬编码的随机种子**，导致类别分配不完全受用户 seed 控制。

## 详细分析

### 1. **类别列表打乱：硬编码 seed=2023（不受用户 seed 影响）**

#### 位置 1：`generate_federated_fewshot_dataset` (第 266-267 行)
```python
class_list = list(range(0, class_num))
random.seed(2023)  # ⚠️ 硬编码，不受用户 seed 影响
random.shuffle(class_list)
```

#### 位置 2：`generate_federated_dataset` (第 400-401 行)
```python
class_list = list(range(0, class_num))
random.seed(2023)  # ⚠️ 硬编码，不受用户 seed 影响
random.shuffle(class_list)
```

**影响**：
- 无论用户传入什么 seed，类别列表的打乱顺序都是固定的（基于 seed=2023）
- 这意味着每个客户端分配到的**类别顺序**是固定的

### 2. **客户端分配：受用户 seed 影响**

#### 位置 1：`generate_federated_fewshot_dataset` (第 288-289 行)
```python
client_list = list(range(0, num_users))
random.shuffle(client_list)  # ✅ 受用户 seed 影响
```

#### 位置 2：`generate_federated_dataset` (第 419-420 行)
```python
client_list = list(range(0, num_users))
random.shuffle(client_list)  # ✅ 受用户 seed 影响
```

**影响**：
- 客户端 ID 的分配顺序受用户 seed 影响
- 但类别列表顺序是固定的，所以最终每个客户端 ID 分配到的类别组合可能不同

### 3. **样本采样：受用户 seed 影响**

#### 位置 1：`generate_federated_fewshot_dataset` (第 333, 346 行)
```python
sampled_items = random.sample(items, num_shots)  # ✅ 受用户 seed 影响
```

#### 位置 2：`generate_federated_dataset` (第 436 行)
```python
random.shuffle(sample_order[label])  # ✅ 受用户 seed 影响
```

**影响**：
- 每个类别内哪些样本被分配给客户端，完全受用户 seed 影响
- 不同 seed 会导致每个客户端得到不同的样本

### 4. **数据下采样：受用户 seed 影响**

#### 位置：`prepare_federated_data` (第 787 行)
```python
rng = random.Random(getattr(cfg, 'SEED', 1))  # ✅ 受用户 seed 影响
train = self.per_class_downsample(train, train_sample_ratio, rng)
test = self.per_class_downsample(test, test_sample_ratio, rng)
```

**影响**：
- 数据下采样时，哪些样本被保留，受用户 seed 影响

## 综合影响分析

### 场景 1：Non-IID，无重叠 (repeat_rate=0.0)

**类别分配逻辑**：
```python
if idx == num_users - 1:
    user_class_dict[idx] = class_list[idx * class_per_user:class_num]
else:
    user_class_dict[idx] = class_list[idx * class_per_user:(idx + 1) * class_per_user]
```

**影响**：
- ❌ **类别分配不受 seed 影响**：因为 `class_list` 的顺序是固定的（seed=2023）
- ✅ **样本分配受 seed 影响**：每个类别内哪些样本分配给客户端，受 seed 影响

**示例**（假设 10 个类别，3 个客户端）：
- Seed=1: Client 0 得到类别 [0,1,2]，Client 1 得到类别 [3,4,5]，Client 2 得到类别 [6,7,8,9]
- Seed=2: **相同的类别分配**（因为 class_list 顺序固定）
- 但每个客户端得到的**具体样本**不同（因为样本采样受 seed 影响）

### 场景 2：Non-IID，有重叠 (repeat_rate > 0)

**类别分配逻辑**：
- 共享类别：`class_repeat_list = class_list[0:repeat_num]`
- 客户端分组：`random.shuffle(client_list)` - **受 seed 影响**

**影响**：
- ❌ **共享类别列表不受 seed 影响**：因为 `class_list` 顺序固定
- ✅ **客户端分组受 seed 影响**：哪些客户端在同一 fold，受 seed 影响
- ✅ **样本分配受 seed 影响**：每个类别内哪些样本分配给客户端，受 seed 影响

**示例**（假设 10 个类别，3 个客户端，repeat_rate=0.3）：
- 共享类别：固定为 `class_list[0:3]`（假设是类别 [2,5,8]）
- Seed=1: Client 0 和 Client 1 在同一 fold，共享类别 [2,5,8]
- Seed=2: Client 0 和 Client 2 在同一 fold，共享类别 [2,5,8]
- 但每个客户端得到的**具体样本**不同

### 场景 3：IID

**类别分配逻辑**：
```python
if is_iid:
    user_class_dict[idx] = list(range(0, class_num))  # 所有客户端都有所有类别
```

**影响**：
- ✅ **类别分配不受 seed 影响**：所有客户端都有所有类别
- ✅ **样本分配受 seed 影响**：每个类别内哪些样本分配给哪个客户端，受 seed 影响

## 问题总结

### 主要问题

1. **类别列表打乱硬编码**：
   - `random.seed(2023)` 硬编码在代码中
   - 导致类别分配顺序不受用户 seed 控制
   - 这会影响实验的可复现性和对比公平性

2. **部分随机性受 seed 控制**：
   - 样本采样受 seed 影响 ✅
   - 客户端分组受 seed 影响 ✅
   - 但类别列表顺序不受 seed 影响 ❌

### 实际影响

**对于 Non-IID 场景**：
- 不同 seed 下，每个客户端分配到的**类别组合可能相同**（因为类别列表顺序固定）
- 但每个客户端得到的**具体样本不同**（因为样本采样受 seed 影响）

**对于 IID 场景**：
- 所有客户端都有所有类别，不受影响
- 但样本分配受 seed 影响

## 建议修复

### 方案 1：使用传入的 seed（推荐）

修改 `generate_federated_fewshot_dataset` 和 `generate_federated_dataset`：

```python
# 当前代码（错误）
random.seed(2023)
random.shuffle(class_list)

# 修改为（正确）
# 使用 prepare_federated_data 传入的 rng，或者从 cfg 获取 SEED
rng = random.Random(getattr(cfg, 'SEED', 1))
class_list = list(range(0, class_num))
rng.shuffle(class_list)
```

### 方案 2：将 rng 作为参数传递

修改函数签名，将 `rng` 作为参数传递：

```python
def generate_federated_dataset(self, *data_sources, rng=None, ...):
    if rng is None:
        rng = random.Random(getattr(cfg, 'SEED', 1))
    
    class_list = list(range(0, class_num))
    rng.shuffle(class_list)  # 使用传入的 rng
```

## 结论

**当前状态**：
- ❌ Seed **不完全**影响类别分配（因为硬编码 seed=2023）
- ✅ Seed **完全**影响样本分配
- ✅ Seed **完全**影响客户端分组（当有 repeat_rate 时）

**修复后**：
- ✅ Seed **完全**影响类别分配
- ✅ Seed **完全**影响样本分配
- ✅ Seed **完全**影响客户端分组

**建议**：修复硬编码的 `random.seed(2023)`，使类别分配也受用户 seed 控制，确保实验的完全可复现性。

