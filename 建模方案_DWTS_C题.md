# 2026 MCM Problem C: Dancing with the Stars 建模方案

## 目录

1. [数据结构概览](#1-数据结构概览)
2. [任务一：观众投票估计模型](#2-任务一观众投票估计模型)
3. [任务二：投票方法对比分析](#3-任务二投票方法对比分析)
4. [任务三：影响因素分析模型](#4-任务三影响因素分析模型)
5. [任务四：新投票系统设计](#5-任务四新投票系统设计)
6. [指标定义体系](#6-指标定义体系)
7. [可视化方案](#7-可视化方案)
8. [代码实现框架](#8-代码实现框架)

---

## 1. 数据结构概览

### 1.1 原始数据说明

| 字段名称 | 类型 | 说明 | 示例值 |
|---------|------|------|--------|
| `celebrity_name` | String | 明星选手姓名 | Jerry Rice |
| `ballroom_partner` | String | 专业舞伴姓名 | Cheryl Burke |
| `celebrity_industry` | Categorical | 选手行业类别 | Athlete, Actor/Actress, Singer/Rapper |
| `celebrity_homestate` | String | 选手家乡州(美国) | Ohio, California |
| `celebrity_homecountry/region` | String | 选手国籍/地区 | United States, England |
| `celebrity_age_during_season` | Integer | 参赛时年龄 | 29, 43, 56 |
| `season` | Integer | 赛季编号 | 1-34 |
| `results` | String | 比赛结果描述 | 1st Place, Eliminated Week 3 |
| `placement` | Integer | 最终名次 | 1, 2, 3, ... |
| `weekX_judgeY_score` | Float/N/A | 第X周第Y位评委打分 | 7, 8.5, N/A, 0 |

### 1.2 数据特征统计

```
总记录数: 422条选手-赛季记录
赛季范围: 第1季 - 第34季
周次范围: 第1周 - 第11周
评委数量: 3-4位/周
得分范围: 1-10分 (含小数, 0表示已淘汰)
行业类别: ~15种 (Athlete, Actor/Actress, Singer/Rapper, TV Personality, Model等)
```

### 1.3 特殊数据处理规则

| 情况 | 原始值 | 处理方式 |
|------|--------|----------|
| 无第4位评委 | N/A | 按3评委计算均分 |
| 已淘汰选手 | 0 | 排除出该周计算 |
| 赛季未进行的周次 | N/A | 不纳入分析 |
| 退赛选手 | Withdrew | 标记为特殊状态，单独分析 |
| 多舞蹈平均分 | 8.5 | 保持原值 |
| 双淘汰周 | - | 根据实际情况确定排名 |

### 1.4 衍生特征工程

```python
# 核心衍生特征
weekly_total_score      = sum(weekX_judge1 + weekX_judge2 + weekX_judge3 [+ weekX_judge4])
weekly_avg_score        = weekly_total_score / num_judges
cumulative_score        = sum(weekly_total_score for all prior weeks)
rank_among_remaining    = rank of weekly_total_score among non-eliminated contestants
score_percentile        = (score - min) / (max - min) within week
improvement_rate        = (current_week_avg - previous_week_avg) / previous_week_avg
consistency_std         = std(all weekly_avg_scores) for the contestant
is_domestic             = 1 if celebrity_homecountry == "United States" else 0
age_group               = categorical(young/middle/senior based on age)
industry_encoded        = one-hot encoding of celebrity_industry
partner_experience      = count of prior seasons for ballroom_partner
elimination_week        = extracted from results string
survived_weeks          = elimination_week - 1 or total_weeks if finalist
```

---

## 2. 任务一：观众投票估计模型

### 2.1 问题形式化

**目标**：估计每位选手每周获得的观众投票数 $V_{i,w}$（未知）

**约束条件**：
- 每周淘汰的选手是综合得分最低者
- 综合得分 = f(评委得分, 观众投票)
- 已知每周实际淘汰结果

### 2.2 模型一：约束优化反推模型 (Constrained Optimization)

#### 2.2.1 排名法下的反推

设第 $w$ 周有 $n$ 位选手，选手 $i$ 的：
- 评委总分：$S_i^{(w)}$（已知）
- 评委排名：$R_i^{J}$（由 $S_i^{(w)}$ 排序得出）
- 观众投票数：$V_i^{(w)}$（待估计）
- 观众排名：$R_i^{F}$（由 $V_i^{(w)}$ 排序得出）
- 综合排名：$R_i^{C} = R_i^{J} + R_i^{F}$

**约束条件**：被淘汰选手 $e$ 满足 $R_e^{C} \geq R_j^{C}$ 对所有 $j \neq e$

**优化目标**：
$$\min \sum_{w} \sum_{i} \left( V_i^{(w)} - \hat{V}_i^{(w)} \right)^2$$

其中 $\hat{V}_i^{(w)}$ 为基于特征的预测值（见2.3节）

#### 2.2.2 百分比法下的反推

设总观众投票数为 $T^{(w)}$，选手 $i$ 的投票占比为 $p_i^{(w)} = V_i^{(w)} / T^{(w)}$

- 评委百分比：$P_i^{J} = S_i^{(w)} / \sum_j S_j^{(w)}$
- 观众百分比：$P_i^{F} = V_i^{(w)} / \sum_j V_j^{(w)}$
- 综合百分比：$P_i^{C} = P_i^{J} + P_i^{F}$

**约束条件**：被淘汰选手 $e$ 满足 $P_e^{C} \leq P_j^{C}$ 对所有 $j \neq e$

### 2.3 模型二：贝叶斯层次模型 (Bayesian Hierarchical Model)

#### 2.3.1 模型结构

```
Level 1 (观众投票生成):
  log(V_{i,w}) ~ Normal(μ_{i,w}, σ²)
  
Level 2 (个体均值建模):
  μ_{i,w} = β₀ + β₁·S_{i,w} + β₂·X_i + β₃·Z_{w} + α_i + γ_s
  
Level 3 (随机效应):
  α_i ~ Normal(0, τ²_α)    # 选手随机效应
  γ_s ~ Normal(0, τ²_γ)    # 赛季随机效应

Prior distributions:
  β ~ Normal(0, 10)
  σ² ~ InverseGamma(1, 1)
  τ² ~ HalfCauchy(0, 5)
```

其中：
- $X_i$：选手特征向量（年龄、行业、国籍等）
- $Z_w$：周次特征（第几周、剩余选手数等）
- $S_{i,w}$：评委得分

#### 2.3.2 先验信息设定

| 参数 | 先验分布 | 理由 |
|------|----------|------|
| β₁ (评委得分系数) | Normal(0.5, 0.2) | 预期评委得分与观众投票正相关 |
| β₂ (行业效应) | Normal(0, 1) | 不同行业影响不确定 |
| σ² | InverseGamma(2, 1) | 弱信息先验 |

### 2.4 模型三：约束回归+MCMC采样

结合约束优化与贝叶斯采样的混合方法：

```python
Algorithm: Constrained MCMC Sampling

1. 初始化: 根据评委得分排名生成初始投票估计
2. For each iteration t = 1, ..., T:
   a. 从后验分布采样参数 θ^(t)
   b. 生成候选投票 V'^(t) ~ q(V | θ^(t), S)
   c. 检验淘汰约束:
      - 若约束满足: 接受 V'^(t)
      - 若约束不满足: 拒绝并重采样
   d. 更新后验
3. 输出: {V^(t)} 的后验样本
```

### 2.5 一致性检验指标

| 指标名称 | 计算公式 | 目标值 |
|----------|----------|--------|
| **淘汰一致率** | $\frac{\text{预测淘汰=实际淘汰的周次}}{\text{总周次}}$ | ≥ 95% |
| **排名一致性** | Kendall's τ between predicted and actual elimination order | ≥ 0.85 |
| **底部二人准确率** | 实际淘汰者在预测底2名的比例 | ≥ 90% |

### 2.6 确定性度量指标

| 指标名称 | 计算公式 | 说明 |
|----------|----------|------|
| **后验标准差** | $\text{Std}(V_{i,w}^{(t)})$ | 越小越确定 |
| **95%置信区间宽度** | $V_{i,w}^{97.5\%} - V_{i,w}^{2.5\%}$ | 越窄越确定 |
| **变异系数** | $\text{CV} = \sigma / \mu$ | 归一化不确定性 |
| **边际敏感度** | $\frac{\partial \text{Elimination}}{\partial V_{i,w}}$ | 投票变化对结果的敏感程度 |
| **确定性分类** | High/Medium/Low based on CV thresholds | CV<0.1: High, 0.1-0.3: Medium, >0.3: Low |

---

## 3. 任务二：投票方法对比分析

### 3.1 两种方法的数学形式

#### 方法A：排名法 (Rank-based, 用于S1-2, S28-34)

$$\text{Combined Score}_i = \text{Rank}_i^{\text{Judge}} + \text{Rank}_i^{\text{Fan}}$$

淘汰规则：$\text{Eliminated} = \arg\max_i \text{Combined Score}_i$

#### 方法B：百分比法 (Percentage-based, 用于S3-27)

$$\text{Combined Score}_i = \frac{S_i}{\sum_j S_j} + \frac{V_i}{\sum_j V_j}$$

淘汰规则：$\text{Eliminated} = \arg\min_i \text{Combined Score}_i$

### 3.2 对比分析框架

```
For each season s = 1, ..., 34:
  For each week w in season s:
    1. 获取实际评委得分 S_{i,w}
    2. 使用任务一估计的观众投票 V_{i,w}
    3. 分别应用排名法和百分比法计算综合得分
    4. 确定两种方法下的淘汰人选
    5. 记录差异
```

### 3.3 对比指标体系

| 指标类别 | 指标名称 | 计算方法 |
|----------|----------|----------|
| **结果差异** | 淘汰差异率 | 两种方法导致不同淘汰结果的周次占比 |
| | 名次差异度 | 最终名次差异的平均绝对值 |
| **权重偏向** | 观众权重指数 | 低评委分选手进入前三的频率 |
| | 评委权重指数 | 高评委分选手被淘汰的频率 |
| **争议程度** | 争议案例数 | 评委排名与最终排名差>3的选手数 |
| | 极端争议率 | 评委得分最低进入前三的比例 |

### 3.4 争议案例深度分析

#### 案例清单
| 赛季 | 选手 | 最终名次 | 争议类型 |
|------|------|----------|----------|
| S2 | Jerry Rice | 2nd | 5周最低评委分仍获亚军 |
| S4 | Billy Ray Cyrus | 5th | 6周最低评委分仍获第5 |
| S11 | Bristol Palin | 3rd | 12次最低评委分仍获季军 |
| S27 | Bobby Bones | 1st | 持续低分仍获冠军 |

#### 分析维度
1. **假设检验**：在另一种方法下的模拟名次
2. **敏感性分析**：需要多少投票变化才能改变结果
3. **评委裁决影响**：底2评委选择机制的作用

### 3.5 评委裁决机制分析 (Bottom-2 Judge Vote)

从S28开始的新规则：底部两名由评委投票决定淘汰谁

**模拟方法**：
```python
def simulate_judge_tiebreaker(bottom_two, judge_scores):
    # 假设评委倾向于保留得分更高者
    if judge_scores[bottom_two[0]] > judge_scores[bottom_two[1]]:
        eliminated = bottom_two[1]
        prob_this_outcome = 0.8  # 假设80%概率
    else:
        eliminated = bottom_two[0]
        prob_this_outcome = 0.8
    return eliminated, prob_this_outcome
```

**分析指标**：
- 评委裁决翻转率：评委选择与纯综合得分不同的比例
- 评委一致性：评委选择与评委得分排名一致的比例

---

## 4. 任务三：影响因素分析模型

### 4.1 分析目标

量化以下因素对比赛表现（评委得分 & 观众投票）的影响：
- **专业舞伴效应**：不同舞伴对选手表现的影响
- **选手特征效应**：年龄、行业、国籍等

### 4.2 模型一：多元回归分析

#### 4.2.1 评委得分模型

$$S_{i,w} = \alpha + \beta_1 \cdot \text{Partner}_i + \beta_2 \cdot \text{Age}_i + \beta_3 \cdot \text{Industry}_i + \beta_4 \cdot \text{Week}_w + \epsilon$$

#### 4.2.2 观众投票模型

$$\log(V_{i,w}) = \gamma_0 + \gamma_1 \cdot \text{Partner}_i + \gamma_2 \cdot \text{Age}_i + \gamma_3 \cdot \text{Industry}_i + \gamma_4 \cdot S_{i,w} + \eta$$

### 4.3 模型二：随机森林+SHAP解释

```python
# 特征集
features = [
    'partner_encoded',           # 舞伴(one-hot或target编码)
    'celebrity_age',             # 年龄
    'industry_encoded',          # 行业
    'is_domestic',               # 是否美国人
    'week_number',               # 周次
    'remaining_contestants',     # 剩余选手数
    'cumulative_judge_score',    # 累计评委得分
    'prior_week_score',          # 上周得分
    'score_improvement',         # 得分提升率
    'partner_experience'         # 舞伴经验(历史参赛次数)
]

# 目标变量
target_judge = 'weekly_avg_judge_score'
target_fan = 'estimated_fan_vote'

# 模型
model_judge = RandomForestRegressor(n_estimators=500, max_depth=10)
model_fan = RandomForestRegressor(n_estimators=500, max_depth=10)

# SHAP解释
explainer_judge = shap.TreeExplainer(model_judge)
shap_values_judge = explainer_judge.shap_values(X_test)
```

### 4.4 模型三：混合效应模型

```
Judge Score Model:
  S_{i,w,s} = β₀ + β₁X_i + β₂Z_w + u_partner[i] + v_season[s] + ε
  u_partner ~ N(0, σ²_partner)
  v_season ~ N(0, σ²_season)

Fan Vote Model:
  log(V_{i,w,s}) = γ₀ + γ₁X_i + γ₂Z_w + γ₃S_{i,w} + α_partner[i] + θ_season[s] + η
```

### 4.5 关键发现框架

| 分析维度 | 评委得分影响 | 观众投票影响 | 差异分析 |
|----------|--------------|--------------|----------|
| **舞伴效应** | 量化各舞伴的"加成" | 量化各舞伴的"人气加成" | 技术vs人气 |
| **年龄效应** | 是否年轻选手更有优势 | 年龄与人气关系 | 评委偏好vs观众偏好 |
| **行业效应** | 不同职业背景的表现差异 | 粉丝基础差异 | 专业能力vs知名度 |
| **国籍效应** | 是否存在国籍偏见 | 主场优势 | 评判公正性 |

### 4.6 舞伴效应量化

```python
# 计算每位舞伴的平均"附加值"
partner_effect = {}
for partner in unique_partners:
    partner_contestants = data[data['ballroom_partner'] == partner]
    # 控制选手自身特征后的残差均值
    partner_effect[partner] = {
        'judge_bonus': mean(residuals_judge),
        'fan_bonus': mean(residuals_fan),
        'win_rate': sum(placement <= 3) / count,
        'avg_placement': mean(placement),
        'experience': count_seasons
    }
```

---

## 5. 任务四：新投票系统设计

### 5.1 设计目标

1. **公平性**：技术优秀的选手不应过早淘汰
2. **观赏性**：保持观众参与度和悬念
3. **简洁性**：规则易于理解和执行

### 5.2 方案一：加权混合法

$$\text{Score}_i = w_J \cdot P_i^J + w_F \cdot P_i^F$$

其中权重随比赛阶段动态调整：

| 阶段 | 周次 | 评委权重 $w_J$ | 观众权重 $w_F$ |
|------|------|----------------|----------------|
| 初期 | 1-3 | 0.6 | 0.4 |
| 中期 | 4-7 | 0.5 | 0.5 |
| 后期 | 8+ | 0.4 | 0.6 |

**理由**：初期侧重技术筛选，后期侧重观众偏好

### 5.3 方案二：淘汰保护机制

```
规则:
1. 若选手评委得分排名前50%，获得"技术保护"
2. 技术保护选手不能在本周被淘汰
3. 淘汰从无保护的选手中按综合得分最低选取
4. 若所有选手都有保护，则取消保护正常淘汰
```

### 5.4 方案三：积分累计制

$$\text{Cumulative Score}_i^{(w)} = \sum_{t=1}^{w} \left( \alpha^{w-t} \cdot \text{Combined}_i^{(t)} \right)$$

其中 $\alpha = 0.8$ 为衰减因子，近期表现权重更高

### 5.5 方案四：多轮投票制

```
Round 1: 确定底部3名 (基于评委+观众综合)
Round 2: 底部3名进行"加赛"表演
Round 3: 观众24小时投票决定淘汰对象
```

### 5.6 新系统评价指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **技术公平性** | 前50%评委分选手被淘汰的比例 | < 20% |
| **观众满意度** (模拟) | 观众投票与最终结果的相关性 | > 0.7 |
| **悬念指数** | 每周淘汰结果的可预测性倒数 | 中等 |
| **争议减少率** | 相比原系统的争议案例减少比例 | > 50% |
| **执行复杂度** | 规则条目数 | < 5条 |

### 5.7 历史数据回测

对每种新方案，使用历史数据进行回测：

```python
def backtest_voting_system(system, historical_data):
    results = []
    for season in seasons:
        for week in weeks:
            # 应用新系统规则
            new_eliminated = system.determine_elimination(
                judge_scores[season][week],
                estimated_votes[season][week]
            )
            # 记录与实际的差异
            results.append({
                'season': season,
                'week': week,
                'original_eliminated': actual_eliminated,
                'new_eliminated': new_eliminated,
                'changed': original != new
            })
    return evaluate_metrics(results)
```

---

## 6. 指标定义体系

### 6.1 模型质量指标

| 指标名称 | 符号 | 公式 | 应用场景 |
|----------|------|------|----------|
| 均方误差 | MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 回归模型评估 |
| 平均绝对误差 | MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | 回归模型评估 |
| 决定系数 | R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释变异比例 |
| 对数似然 | LL | $\sum \log P(y|\theta)$ | 贝叶斯模型比较 |
| DIC/WAIC | - | 贝叶斯信息准则 | 模型复杂度惩罚 |

### 6.2 预测准确性指标

| 指标名称 | 公式 | 目标值 |
|----------|------|--------|
| 淘汰预测准确率 | $\frac{\sum \mathbb{1}[\hat{e}=e]}{\text{total weeks}}$ | ≥ 95% |
| Top-3预测准确率 | $\frac{\sum \mathbb{1}[\hat{T}_3 \cap T_3 \neq \emptyset]}{3}$ | ≥ 80% |
| 排名相关系数 | Spearman's ρ | ≥ 0.85 |

### 6.3 不确定性量化指标

| 指标名称 | 公式 | 说明 |
|----------|------|------|
| 置信区间覆盖率 | $\frac{\sum \mathbb{1}[y \in CI]}{n}$ | 应接近95% |
| 区间平均宽度 | $\frac{1}{n}\sum(CI_{upper} - CI_{lower})$ | 越窄越好 |
| 校准误差 | $|P_{predicted} - P_{observed}|$ | 应趋近0 |

### 6.4 方法对比指标

| 指标名称 | 计算方法 | 解释 |
|----------|----------|------|
| 结果一致率 | 两方法相同淘汰的比例 | 高=方法差异小 |
| 观众偏好度 | 低评委分选手晋级的频率 | 高=偏向观众 |
| 评委偏好度 | 高评委分选手晋级的频率 | 高=偏向评委 |
| Gini系数 | 综合得分分布不均匀度 | 反映竞争激烈程度 |

---

## 7. 可视化方案

### 7.1 数据探索性可视化

```
图1: 评委得分分布热力图
- X轴: 周次 (Week 1-11)
- Y轴: 赛季 (Season 1-34)
- 颜色: 平均评委得分
- 目的: 展示评分随时间的整体趋势

图2: 选手行业分布饼图/条形图
- 各行业选手数量
- 各行业平均名次
- 目的: 了解数据构成

图3: 评委得分箱线图
- 按行业分组
- 展示得分分布差异
- 目的: 初步了解行业效应

图4: 选手年龄与名次散点图
- X轴: 年龄
- Y轴: 最终名次
- 颜色: 行业
- 目的: 探索年龄效应
```

### 7.2 模型结果可视化

```
图5: 观众投票估计置信区间图
- X轴: 选手
- Y轴: 估计投票数 (对数刻度)
- 误差棒: 95% CI
- 分组: 按周次
- 目的: 展示估计及不确定性

图6: 淘汰预测准确性混淆矩阵
- 预测淘汰 vs 实际淘汰
- 颜色深浅: 频次
- 目的: 评估模型准确性

图7: 两种投票方法结果对比
- 双轴图: 排名法得分 vs 百分比法得分
- 标记差异点
- 目的: 直观对比两种方法

图8: 争议案例轨迹图
- X轴: 周次
- Y轴: 综合排名
- 多线: 评委排名、观众排名、综合排名
- 选手: Jerry Rice, Bristol Palin等
- 目的: 深入分析争议产生原因
```

### 7.3 影响因素可视化

```
图9: SHAP特征重要性瀑布图
- 各特征对评委得分的影响
- 各特征对观众投票的影响
- 目的: 解释模型

图10: 舞伴效应雷达图
- 各舞伴在多个维度的表现
- 维度: 冠军率、平均名次、评委加成、观众加成
- 目的: 全面评估舞伴价值

图11: 行业效应对比条形图
- 横向: 对评委得分影响
- 纵向: 对观众投票影响
- 分组: 各行业
- 目的: 比较行业在两方面的差异

图12: 交互效应热力图
- X轴: 年龄组
- Y轴: 行业
- 颜色: 平均名次
- 目的: 探索特征交互效应
```

### 7.4 新系统评估可视化

```
图13: 各方案回测结果对比
- 雷达图: 公平性、观赏性、简洁性等多维度
- 目的: 综合评价各方案

图14: 争议减少效果图
- 柱状图: 原系统 vs 新系统的争议案例数
- 按赛季分组
- 目的: 量化改进效果

图15: 模拟冠军分布变化
- 堆叠条形图: 各方案下不同类型选手获冠军的比例
- 目的: 评估系统公平性

图16: 悬念曲线
- X轴: 周次
- Y轴: 淘汰可预测性 (熵)
- 多线: 各方案
- 目的: 评估观赏性
```

### 7.5 报告专用可视化

```
图17: 研究框架流程图
- 数据 → 模型 → 分析 → 建议
- 清晰展示研究逻辑

图18: 关键发现信息图
- 图标+数字形式
- 突出核心结论

图19: 建议方案对比表格
- 各维度得分
- 最终推荐标识
```

---

## 8. 代码实现框架

### 8.1 项目结构

```
DWTS_Analysis/
├── data/
│   ├── raw/
│   │   └── 2026_MCM_Problem_C_Data.csv
│   └── processed/
│       ├── cleaned_data.csv
│       ├── features.csv
│       └── estimated_votes.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── vote_estimation.py
│   ├── method_comparison.py
│   ├── factor_analysis.py
│   ├── new_system_design.py
│   └── visualization.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Vote_Estimation.ipynb
│   ├── 03_Method_Comparison.ipynb
│   ├── 04_Factor_Analysis.ipynb
│   └── 05_New_System.ipynb
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
├── requirements.txt
└── README.md
```

### 8.2 核心代码模块

#### 8.2.1 数据预处理

```python
# src/data_preprocessing.py
import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """加载并清洗原始数据"""
    df = pd.read_csv(filepath)
    
    # 处理N/A值
    df = df.replace('N/A', np.nan)
    
    # 提取淘汰周次
    df['elimination_week'] = df['results'].apply(extract_elimination_week)
    
    # 计算每周总分
    for week in range(1, 12):
        cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        df[f'week{week}_total'] = df[cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce').sum(), axis=1
        )
        df[f'week{week}_avg'] = df[cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean(), axis=1
        )
    
    return df

def create_weekly_data(df):
    """转换为周级别长格式数据"""
    records = []
    for _, row in df.iterrows():
        for week in range(1, 12):
            if pd.notna(row[f'week{week}_judge1_score']) and row[f'week{week}_judge1_score'] != 0:
                records.append({
                    'celebrity': row['celebrity_name'],
                    'partner': row['ballroom_partner'],
                    'season': row['season'],
                    'week': week,
                    'total_score': row[f'week{week}_total'],
                    'avg_score': row[f'week{week}_avg'],
                    'final_placement': row['placement'],
                    'industry': row['celebrity_industry'],
                    'age': row['celebrity_age_during_season'],
                    'is_domestic': row['celebrity_homecountry/region'] == 'United States'
                })
    return pd.DataFrame(records)
```

#### 8.2.2 投票估计模型

```python
# src/vote_estimation.py
import numpy as np
from scipy.optimize import minimize
import pymc3 as pm

class VoteEstimator:
    def __init__(self, method='bayesian'):
        self.method = method
        self.estimates = None
        self.uncertainty = None
    
    def fit_constrained_optimization(self, scores, eliminated, method='rank'):
        """约束优化方法"""
        n_contestants = len(scores)
        
        def objective(votes):
            # 最小化与先验估计的差异
            prior = self._compute_prior(scores)
            return np.sum((votes - prior) ** 2)
        
        def elimination_constraint(votes):
            # 确保被淘汰者综合得分最低
            combined = self._combine_scores(scores, votes, method)
            eliminated_idx = np.argmin(combined)
            return eliminated_idx - eliminated  # 应为0
        
        # 优化
        result = minimize(
            objective,
            x0=self._initial_votes(scores),
            constraints={'type': 'eq', 'fun': elimination_constraint},
            bounds=[(0, None)] * n_contestants
        )
        
        return result.x
    
    def fit_bayesian(self, data):
        """贝叶斯层次模型"""
        with pm.Model() as model:
            # 先验
            beta_score = pm.Normal('beta_score', mu=0.5, sd=0.2)
            beta_age = pm.Normal('beta_age', mu=0, sd=0.1)
            sigma = pm.HalfCauchy('sigma', beta=2)
            
            # 随机效应
            partner_effect = pm.Normal('partner_effect', 
                                       mu=0, sd=1, 
                                       shape=n_partners)
            
            # 线性预测
            mu = (beta_score * data['score'] + 
                  beta_age * data['age'] + 
                  partner_effect[data['partner_idx']])
            
            # 似然
            votes = pm.Lognormal('votes', mu=mu, sd=sigma, observed=None)
            
            # 采样
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
        
        return trace
```

#### 8.2.3 方法对比分析

```python
# src/method_comparison.py

def apply_rank_method(judge_scores, fan_votes):
    """排名法计算综合得分"""
    judge_ranks = np.argsort(np.argsort(-judge_scores)) + 1
    fan_ranks = np.argsort(np.argsort(-fan_votes)) + 1
    combined = judge_ranks + fan_ranks
    return combined

def apply_percentage_method(judge_scores, fan_votes):
    """百分比法计算综合得分"""
    judge_pct = judge_scores / judge_scores.sum()
    fan_pct = fan_votes / fan_votes.sum()
    combined = judge_pct + fan_pct
    return combined

def compare_methods(data, estimated_votes):
    """对比两种方法的结果"""
    results = []
    
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        
        for week in season_data['week'].unique():
            week_data = season_data[season_data['week'] == week]
            
            scores = week_data['total_score'].values
            votes = estimated_votes[(season, week)]
            
            rank_result = apply_rank_method(scores, votes)
            pct_result = apply_percentage_method(scores, votes)
            
            eliminated_rank = np.argmax(rank_result)
            eliminated_pct = np.argmin(pct_result)
            
            results.append({
                'season': season,
                'week': week,
                'eliminated_rank': week_data.iloc[eliminated_rank]['celebrity'],
                'eliminated_pct': week_data.iloc[eliminated_pct]['celebrity'],
                'same_result': eliminated_rank == eliminated_pct
            })
    
    return pd.DataFrame(results)
```

#### 8.2.4 可视化模块

```python
# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_vote_estimates_with_ci(estimates, lower, upper, contestants, week):
    """绘制投票估计及置信区间"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(contestants))
    ax.bar(x, estimates, yerr=[estimates-lower, upper-estimates], 
           capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(contestants, rotation=45, ha='right')
    ax.set_ylabel('Estimated Votes')
    ax.set_title(f'Week {week} Vote Estimates with 95% CI')
    
    plt.tight_layout()
    return fig

def plot_controversy_trajectory(contestant_data, contestant_name):
    """绘制争议选手轨迹图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weeks = contestant_data['week']
    ax.plot(weeks, contestant_data['judge_rank'], 'o-', label='Judge Rank')
    ax.plot(weeks, contestant_data['fan_rank'], 's-', label='Fan Rank (Est.)')
    ax.plot(weeks, contestant_data['combined_rank'], '^-', label='Combined Rank')
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Rank (Lower = Better)')
    ax.set_title(f'Performance Trajectory: {contestant_name}')
    ax.legend()
    ax.invert_yaxis()
    
    return fig

def plot_shap_summary(shap_values, features, title):
    """SHAP特征重要性图"""
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.title(title)
    return fig
```

### 8.3 依赖环境

```
# requirements.txt
pandas>=1.4.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pymc3>=3.11.0  # 或 pymc>=5.0.0
arviz>=0.11.0
shap>=0.40.0
statsmodels>=0.13.0
jupyter>=1.0.0
```

---

## 附录：时间规划建议

| 阶段 | 时间 | 任务 |
|------|------|------|
| Day 1 上午 | 4h | 数据预处理+EDA+特征工程 |
| Day 1 下午 | 4h | 任务一：投票估计模型开发 |
| Day 2 上午 | 4h | 任务一：模型验证+任务二：方法对比 |
| Day 2 下午 | 4h | 任务三：影响因素分析 |
| Day 3 上午 | 4h | 任务四：新系统设计+回测 |
| Day 3 下午 | 4h | 可视化+论文撰写+备忘录 |
| Day 4 | 4h | 润色+检查+提交 |

---

## 附录：关键公式速查

### 排名法
$$\text{Combined Rank}_i = \text{Rank}^J_i + \text{Rank}^F_i$$

### 百分比法
$$\text{Combined \%}_i = \frac{S_i}{\sum S} + \frac{V_i}{\sum V}$$

### 贝叶斯投票估计
$$\log(V_{i,w}) \sim \mathcal{N}(\beta_0 + \beta_1 S_{i,w} + \beta_2 X_i + \alpha_{\text{partner}}, \sigma^2)$$

### 一致性指标
$$\text{Accuracy} = \frac{|\{\text{correct elimination predictions}\}|}{|\{\text{all weeks}\}|}$$

### 确定性指标
$$\text{CV}_{i,w} = \frac{\sigma_{V_{i,w}}}{\mu_{V_{i,w}}}$$

---

*文档生成时间: 2026年1月30日*
*适用于: 2026 MCM Problem C - Dancing with the Stars*
