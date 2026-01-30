# 淘汰判定与NaN处理机制说明

## 1. 淘汰判定机制

### 1.1 两种投票方法的不同淘汰规则

#### 排名法 (Rank-based Method) - S1-2, S28-34
```python
# 计算综合排名（数值越大越差）
judge_ranks = 评委排名  # 评分越高排名越好(rank=1,2,3...)
fan_ranks = 观众排名    # 票数越高排名越好(rank=1,2,3...)
combined_rank = judge_ranks + fan_ranks  # 相加

# 淘汰：综合排名最大者（最差）
eliminated = argmax(combined_rank)
```

**示例**：
| 选手 | 评分 | 评委排名 | 票数 | 观众排名 | 综合排名 | 结果 |
|------|------|---------|------|---------|---------|------|
| A | 28 | 1 | 50000 | 1 | 2 | 安全 |
| B | 25 | 2 | 45000 | 2 | 4 | 安全 |
| C | 22 | 3 | 40000 | 3 | **6** | **淘汰** |

#### 百分比法 (Percentage-based Method) - S3-27
```python
# 计算综合百分比（数值越大越好）
judge_pct = 评委得分 / 总评分  # 占比
fan_pct = 观众投票 / 总投票    # 占比
combined_pct = judge_pct + fan_pct  # 相加

# 淘汰：综合百分比最小者（最差）
eliminated = argmin(combined_pct)
```

**示例**：
| 选手 | 评分 | 评委占比 | 票数 | 观众占比 | 综合占比 | 结果 |
|------|------|---------|------|---------|---------|------|
| A | 28 | 0.373 | 50000 | 0.370 | 0.743 | 安全 |
| B | 25 | 0.333 | 45000 | 0.333 | 0.666 | 安全 |
| C | 22 | 0.293 | 40000 | 0.296 | **0.589** | **淘汰** |

### 1.2 代码实现位置

在 `src/consistency_check.py` 第67-75行：

```python
# 计算预测淘汰者
if use_method == 'rank':
    combined = compute_rank_combined_score(scores, votes)
    pred_idx = get_eliminated_index_rank(combined)  # argmax
else:
    combined = compute_percent_combined_score(scores, votes)
    pred_idx = get_eliminated_index_percent(combined)  # argmin
```

在 `src/utils.py` 第168-190行：

```python
def get_eliminated_index_rank(combined_ranks: np.ndarray) -> int:
    """排名法：综合排名最高（最差）者淘汰"""
    return int(np.argmax(combined_ranks))

def get_eliminated_index_percent(combined_pcts: np.ndarray) -> int:
    """百分比法：综合百分比最低者淘汰"""
    return int(np.argmin(combined_pcts))
```

---

## 2. NaN值处理机制

### 2.1 评委评分中的NaN处理

**来源**：选手在某周被淘汰后，后续周次的评分为N/A或空值

#### 数据清洗阶段
在 `src/data_preprocessing.py` 第54-56行：

```python
# 处理评委得分列，将N/A转换为NaN
score_columns = [col for col in df.columns if 'judge' in col and 'score' in col]
for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

- `pd.to_numeric(..., errors='coerce')`：将无效值（'N/A'、空字符串等）转换为 `NaN`

#### 计算每周总分和平均分
在 `src/utils.py` 第54-77行和第82-106行：

```python
def compute_weekly_total_score(row: pd.Series, week: int) -> float:
    """计算某周的评委总分"""
    total = 0
    count = 0
    
    for judge in range(1, 5):  # 4位评委
        col = f'week{week}_judge{judge}_score'
        if col in row.index:
            val = row[col]
            # 关键：检查是否为有效值
            if pd.notna(val) and val != 'N/A':
                try:
                    score = float(val)
                    if score > 0:  # 排除已淘汰的0分
                        total += score
                        count += 1
                except (ValueError, TypeError):
                    pass
    
    return total if count > 0 else 0
```

**处理逻辑**：
1. **跳过NaN值**：`pd.notna(val)` 检查，NaN不参与求和
2. **只统计有效评分**：`count` 记录有效评委数
3. **返回有效评分之和**：不是除以4，而是直接返回总分

**示例**：
```
评委1: 8    ✓ 有效
评委2: 8    ✓ 有效  
评委3: NaN  ✗ 跳过
评委4: 7    ✓ 有效

total_score = 8 + 8 + 7 = 23  （不是 23/3 = 7.67）
```

### 2.2 平均分计算

在 `compute_weekly_avg_score()` 函数中：

```python
def compute_weekly_avg_score(row: pd.Series, week: int) -> float:
    """计算某周的评委平均分"""
    scores = []
    
    for judge in range(1, 5):
        col = f'week{week}_judge{judge}_score'
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val != 'N/A':
                try:
                    score = float(val)
                    if score > 0:
                        scores.append(score)  # 收集有效分数
                except (ValueError, TypeError):
                    pass
    
    return np.mean(scores) if scores else 0  # 有效分数的平均
```

**处理逻辑**：
- 只收集有效的评委评分到列表中
- 计算平均分 = 有效评分之和 / 有效评委数

**示例**：
```
评委1: 8
评委2: 8  
评委3: NaN  ← 跳过
评委4: 7

avg_score = (8 + 8 + 7) / 3 = 7.67
```

### 2.3 淘汰周次中的NaN处理

在 `get_elimination_info()` 函数中（第179-181行）：

```python
elim_week = row['elimination_week']
# 检查 elim_week 是有效数字（不是 None 也不是 NaN）
if elim_week is not None and pd.notna(elim_week):
    elim_week = int(elim_week)  # 确保是整数
    # ... 后续处理
```

**处理逻辑**：
- 决赛选手（Winner/2nd/3rd Place）：`elimination_week = None`
- 退赛选手（Withdrew）：`elimination_week = None`
- 这些选手**不会被包含在淘汰记录中**，只记录正常淘汰的选手

---

## 3. 关键设计决策总结

| 场景 | 处理方式 | 原因 |
|------|---------|------|
| 评委评分为N/A | 跳过，不参与计算 | 已淘汰选手不再有评分 |
| 总分计算 | 有效评分之和 | 保留原始分数总和 |
| 平均分计算 | 有效评分的平均 | 反映实际平均水平 |
| 淘汰周次为None | 跳过该选手 | 决赛选手/退赛不算"被淘汰" |
| 排名法淘汰 | 综合排名最大（argmax） | 排名越大越差 |
| 百分比法淘汰 | 综合百分比最小（argmin） | 百分比越小越差 |

---

## 4. 数据流示意图

```
原始CSV数据
    ↓
[pd.to_numeric(errors='coerce')]  ← 将 'N/A' → NaN
    ↓
清洗后数据
    ↓
[compute_weekly_total_score]
    ├─ pd.notna(val) → True: 累加
    └─ pd.notna(val) → False: 跳过
    ↓
total_score (有效评分之和)
    ↓
[建模使用]
    ↓
投票估计
    ↓
[一致性检验]
    ├─ 排名法: argmax(judge_rank + fan_rank)
    └─ 百分比法: argmin(judge_pct + fan_pct)
    ↓
预测淘汰者
```

---

## 5. 常见问题

**Q1: 为什么不是平均分？**
A: 数据中使用的是**总分**（sum of valid scores），不是平均分。这是节目的实际规则。

**Q2: 如果某周只有2位评委打分怎么办？**
A: 只累加这2位评委的分数，不会除以4。例如：8+7=15。

**Q3: 决赛选手会被预测淘汰吗？**
A: 不会，因为他们的 `elimination_week = None`，不会出现在淘汰记录中。

**Q4: 如果所有评委都是NaN会怎样？**
A: `total_score = 0`，该选手该周的记录会被过滤掉（见 `create_weekly_long_format()` 第115行）。

---

*此文档解释了淘汰判定和NaN处理的完整逻辑*
