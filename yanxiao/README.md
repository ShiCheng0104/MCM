# 🌟 DWTS 观众投票估计模型 (Problem C - Task 1)

> **2026 MCM Problem C: Data With The Stars**  
> 本项目实现了对《与星共舞》(Dancing with the Stars) 节目中观众投票数的数学建模与估计。

---

## 📋 问题背景

《与星共舞》是美国著名真人秀节目，观众投票与评委评分共同决定选手的去留。然而，**观众投票数从未公开**，这为数据分析带来了挑战。

本项目的目标是：
- 开发数学模型估算每位选手在每周获得的观众投票数
- 量化估计结果的不确定性
- 验证模型与实际淘汰结果的一致性

### 投票规则演变

| 赛季 | 计分方法 | 说明 |
|:----:|:--------:|------|
| S1-S2 | 排名法 (Rank-based) | 评委排名 + 观众排名 |
| S3-S27 | 百分比法 (Percentage-based) | 评委得分占比 + 观众票数占比 |
| S28-S34 | 排名法 + 评委投票 | 综合排名最低两人由评委投票决定 |

---

## 🏗️ 项目结构

```
yanxiao/
├── 📄 README.md                     # 项目说明文档
├── 🚀 main.py                       # 主程序入口
├── ⚙️ config.py                     # 配置参数模块
├── 📋 requirements.txt              # Python依赖
│
├── 📁 data/                         # 数据模块
│   └── __init__.py
│
├── 📁 src/                          # 核心源代码
│   ├── __init__.py
│   ├── data_preprocessing.py       # 数据预处理
│   ├── vote_estimator.py           # 投票估计器（整合多模型）
│   ├── consistency_check.py        # 一致性检验
│   ├── uncertainty_measure.py      # 不确定性度量
│   └── utils.py                    # 工具函数库
│
├── 📁 models/                       # 数学模型
│   ├── __init__.py
│   ├── baseline_model.py           # 基线模型
│   ├── constrained_optimization.py # 约束优化模型
│   └── bayesian_model.py           # 贝叶斯层次模型
│
├── 📁 visualization/                # 可视化模块
│   ├── __init__.py
│   └── plots.py                    # 绑图函数
│
└── 📁 outputs/                      # 输出结果
    ├── vote_estimates.csv          # 投票估计结果
    ├── consistency_results.csv     # 一致性检验结果
    └── figures/                    # 可视化图表
```

---

## 🔧 安装与运行

### 环境要求

- Python 3.9+
- 推荐使用虚拟环境

### 安装步骤

```bash
# 1. 进入项目目录
cd d:/竞赛/美赛/yanxiao

# 2. 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行主程序
python main.py
```

### 依赖包说明

| 包名 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `numpy` | 数值计算 |
| `scipy` | 优化算法 |
| `pymc` | 贝叶斯推断 (MCMC) |
| `arviz` | 贝叶斯诊断与可视化 |
| `matplotlib` / `seaborn` | 可视化 |

---

## �️ 数据处理详解

### 原始数据结构

数据文件 `2026_MCM_Problem_C_Data.csv` 包含 **422 条选手记录**，跨越 **34 个赛季**。

| 字段 | 类型 | 说明 |
|------|------|------|
| `celebrity_name` | string | 明星选手姓名 |
| `ballroom_partner` | string | 专业舞伴姓名 |
| `celebrity_industry` | string | 选手所属行业（演员、运动员、歌手等） |
| `celebrity_age_during_season` | int | 参赛时年龄 |
| `celebrity_homecountry/region` | string | 选手国籍/地区 |
| `season` | int | 赛季编号 (1-34) |
| `results` | string | 比赛结果（如 "Eliminated Week 3", "1st Place"） |
| `placement` | int | 最终名次 |
| `week{n}_judge{m}_score` | float | 第n周第m位评委的打分 |

### 数据预处理流程

```
原始CSV ──→ 数据清洗 ──→ 特征工程 ──→ 格式转换 ──→ 模型输入
```

#### Step 1: 数据加载与类型转换

```python
# 加载原始数据
df = pd.read_csv(data_path)

# 将评委得分列转换为数值类型
# 原始数据中 "N/A" 表示无数据（无第4评委或已淘汰）
score_columns = [col for col in df.columns if 'judge' in col and 'score' in col]
for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # "N/A" → NaN
```

#### Step 2: 派生特征提取

从 `results` 字段解析关键信息：

```python
def extract_elimination_week(result_str):
    """
    解析淘汰周次
    - "Eliminated Week 3" → 3
    - "1st Place" → None (决赛选手)
    - "Withdrew Week 5" → None (退赛选手)
    """
    if 'Place' in result_str:
        return None  # 决赛选手
    if 'Withdrew' in result_str:
        return None  # 退赛
    match = re.search(r'Week (\d+)', result_str)
    return int(match.group(1)) if match else None

# 添加派生特征
df['elimination_week'] = df['results'].apply(extract_elimination_week)
df['is_finalist'] = df['results'].str.contains('Place', na=False)
df['is_withdrew'] = df['results'].str.contains('Withdrew', na=False)
df['is_domestic'] = df['celebrity_homecountry/region'] == 'United States'
```

#### Step 3: 周得分聚合

计算每周的评委总分和平均分：

```python
def compute_weekly_total_score(row, week):
    """
    计算第week周的评委总分
    - 跳过 NaN 和 0（已淘汰的标记）
    - 处理多舞平均分（小数值）
    """
    total = 0
    for judge in range(1, 5):  # 最多4位评委
        col = f'week{week}_judge{judge}_score'
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val > 0:
                total += float(val)
    return total

# 为每周计算得分
for week in range(1, 12):  # 最多11周
    df[f'week{week}_total'] = df.apply(
        lambda row: compute_weekly_total_score(row, week), axis=1
    )
```

#### Step 4: 长格式转换

将宽表转换为周级别的长表，便于模型处理：

```python
# 原始: 每行一个选手，包含所有周的得分
# 转换: 每行一个选手在一周的数据

records = []
for _, row in df.iterrows():
    for week in range(1, 12):
        total_score = row[f'week{week}_total']
        if total_score > 0:  # 只保留有效得分的周次
            records.append({
                'celebrity_name': row['celebrity_name'],
                'ballroom_partner': row['ballroom_partner'],
                'celebrity_industry': row['celebrity_industry'],
                'celebrity_age': row['celebrity_age_during_season'],
                'season': row['season'],
                'week': week,
                'total_score': total_score,
                'final_placement': row['placement'],
                'elimination_week': row['elimination_week']
            })

weekly_data = pd.DataFrame(records)
# 结果: ~2500 条周级别记录
```

#### Step 5: 赛季-周次分组

创建便于查询的数据结构：

```python
# 字典: (season, week) → 该周所有活跃选手的DataFrame
season_week_data = {}
for (season, week), group in weekly_data.groupby(['season', 'week']):
    season_week_data[(season, week)] = group.copy()

# 示例: season_week_data[(27, 9)] 返回第27季第9周的所有选手数据
```

### 数据特殊值处理

| 原始值 | 含义 | 处理方式 |
|--------|------|----------|
| `N/A` | 无第4评委或数据缺失 | 转为 NaN，计算时跳过 |
| `0` | 选手已被淘汰 | 跳过该周数据 |
| 小数 (如 `8.5`) | 多舞表演的平均分 | 直接使用 |
| 空白 | 数据缺失 | 转为 NaN |

### 淘汰信息提取

```python
# 提取每周的淘汰记录
elimination_info = []
for _, row in df.iterrows():
    elim_week = row['elimination_week']
    if elim_week is not None:
        elimination_info.append({
            'season': row['season'],
            'week': elim_week,
            'eliminated_name': row['celebrity_name'],
            'eliminated_placement': row['placement'],
            'final_score': row[f'week{elim_week}_total']
        })

elimination_df = pd.DataFrame(elimination_info)
# 结果: ~380 条淘汰记录
```

---

## 📊 数学模型详解

### 模型1: 基线模型 (Baseline Model)

**核心思想**：观众投票与评委得分成正比

$$V_i \propto S_i^{\alpha}$$

其中：
- $V_i$：选手 $i$ 的投票数
- $S_i$：选手 $i$ 的评委总分
- $\alpha$：影响系数（通过网格搜索优化）

#### 实现逻辑

```python
def estimate_votes(judge_scores, alpha=1.0, total_votes=1e6):
    """
    基线投票估计
    
    Args:
        judge_scores: 评委得分数组 [S_1, S_2, ..., S_n]
        alpha: 幂次参数
        total_votes: 假设的总投票数
    
    Returns:
        估计投票数组 [V_1, V_2, ..., V_n]
    """
    # Step 1: 计算基础投票（得分的α次幂）
    base_votes = np.power(judge_scores, alpha)
    
    # Step 2: 归一化到总票数
    votes = base_votes / base_votes.sum() * total_votes
    
    return votes
```

#### 参数优化

通过网格搜索找到最优 $\alpha$：

```python
def fit_alpha(season_week_data, elimination_info):
    """
    网格搜索最优alpha
    目标: 最大化淘汰预测准确率
    """
    alphas = np.arange(0.5, 2.1, 0.1)  # 搜索范围
    best_alpha, best_accuracy = 1.0, 0.0
    
    for alpha in alphas:
        accuracy = compute_elimination_accuracy(alpha, season_week_data, elimination_info)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
    
    return best_alpha  # 通常在 0.8-1.2 之间
```

**优点**：简单直观，计算快速  
**缺点**：无法捕捉评委与观众偏好差异

---

### 模型2: 约束优化模型 (Constrained Optimization)

**核心思想**：利用淘汰结果作为约束条件，反推满足约束的投票分布

#### 数学形式化

**决策变量**：$\mathbf{V} = [V_1, V_2, \ldots, V_n]^T$（各选手投票数）

**目标函数**（最小化与先验的偏差）：
$$\min_{\mathbf{V}} \sum_i \left( \log V_i - \log V_i^{\text{prior}} \right)^2$$

其中先验 $V_i^{\text{prior}}$ 基于评委得分：
$$V_i^{\text{prior}} = \frac{\exp(\lambda \cdot S_i / S_{\max})}{\sum_j \exp(\lambda \cdot S_j / S_{\max})} \times N_{\text{total}}$$

#### 约束条件

**排名法赛季 (S1-S2, S28-S34)**：

设 $k$ 为被淘汰选手，$R_S(i)$ 为选手 $i$ 的评委排名，$R_V(i)$ 为投票排名：

$$R_S(k) + R_V(k) \geq R_S(j) + R_V(j), \quad \forall j \neq k$$

即被淘汰者的综合排名得分（越高越差）必须最大。

**百分比法赛季 (S3-S27)**：

$$\frac{S_k}{\sum_i S_i} + \frac{V_k}{\sum_i V_i} \leq \frac{S_j}{\sum_i S_i} + \frac{V_j}{\sum_i V_i}, \quad \forall j \neq k$$

即被淘汰者的综合百分比必须最小。

#### 实现逻辑

```python
def estimate_votes_rank_method(judge_scores, eliminated_idx, total_votes=1e6):
    """
    排名法约束优化
    
    Args:
        judge_scores: 评委得分数组
        eliminated_idx: 被淘汰选手的索引
        total_votes: 总投票数
    """
    n = len(judge_scores)
    prior = compute_prior_votes(judge_scores, total_votes)
    
    # 目标函数: 最小化对数空间偏差
    def objective(votes):
        log_votes = np.log(votes + 1)
        log_prior = np.log(prior + 1)
        return np.sum((log_votes - log_prior) ** 2)
    
    # 约束: 被淘汰者综合排名最差
    def elimination_constraint(votes):
        # 计算综合排名得分 (越高越差)
        combined = compute_rank_combined_score(judge_scores, votes)
        eliminated_score = combined[eliminated_idx]
        max_other_score = max(combined[i] for i in range(n) if i != eliminated_idx)
        return eliminated_score - max_other_score  # 必须 >= 0
    
    # 边界约束
    bounds = [(100, total_votes * 0.8) for _ in range(n)]
    
    # 等式约束: 总票数
    constraints = [
        {'type': 'ineq', 'fun': elimination_constraint},
        {'type': 'eq', 'fun': lambda v: np.sum(v) - total_votes}
    ]
    
    # SLSQP优化
    result = minimize(objective, prior, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x
```

#### 综合得分计算

```python
def compute_rank_combined_score(judge_scores, fan_votes):
    """
    排名法综合得分 (越高越差，最高者被淘汰)
    """
    # 评委排名 (得分最高 → 排名1)
    judge_ranks = np.argsort(np.argsort(-judge_scores)) + 1
    # 投票排名 (票数最高 → 排名1)
    fan_ranks = np.argsort(np.argsort(-fan_votes)) + 1
    # 综合 = 两排名之和
    return judge_ranks + fan_ranks

def compute_percent_combined_score(judge_scores, fan_votes):
    """
    百分比法综合得分 (越高越好，最低者被淘汰)
    """
    judge_pct = judge_scores / judge_scores.sum()
    fan_pct = fan_votes / fan_votes.sum()
    return judge_pct + fan_pct
```

**优化方法**：SLSQP (Sequential Least Squares Programming)

**优点**：保证与淘汰结果一致  
**缺点**：解可能不唯一

---

### 模型3: 贝叶斯层次模型 (Bayesian Hierarchical Model)

**核心思想**：使用完整的贝叶斯推断，建模选手、舞伴、赛季等多层次随机效应

#### 模型结构

**观测层**：
$$\log(V_{i,w}) \sim \mathcal{N}(\mu_{i,w}, \sigma^2)$$

**线性预测器**：
$$\mu_{i,w} = \beta_0 + \beta_1 \cdot \tilde{S}_{i,w} + \beta_2 \cdot \tilde{A}_i + \alpha_{p[i]} + \gamma_{s[i]} + \delta_{d[i]}$$

其中：
- $\tilde{S}_{i,w} = (S_{i,w} - \bar{S}) / \sigma_S$：标准化评委得分
- $\tilde{A}_i = (A_i - \bar{A}) / \sigma_A$：标准化年龄
- $\alpha_{p[i]}$：选手 $i$ 的舞伴 $p[i]$ 的随机效应
- $\gamma_{s[i]}$：赛季 $s[i]$ 的随机效应
- $\delta_{d[i]}$：行业 $d[i]$ 的随机效应

#### 先验分布

| 参数 | 先验 | 说明 |
|------|------|------|
| $\beta_0$ | $\mathcal{N}(10, 2)$ | 截距（对数尺度） |
| $\beta_1$ | $\mathcal{N}(0, 0.5)$ | 得分系数 |
| $\beta_2$ | $\mathcal{N}(0, 0.1)$ | 年龄系数 |
| $\sigma$ | $\text{HalfNormal}(1)$ | 残差标准差 |
| $\sigma_p$ | $\text{HalfNormal}(0.5)$ | 舞伴效应标准差 |
| $\sigma_s$ | $\text{HalfNormal}(0.3)$ | 赛季效应标准差 |
| $\sigma_d$ | $\text{HalfNormal}(0.5)$ | 行业效应标准差 |
| $\alpha_{p}$ | $\mathcal{N}(0, \sigma_p)$ | 舞伴随机效应 |
| $\gamma_{s}$ | $\mathcal{N}(0, \sigma_s)$ | 赛季随机效应 |
| $\delta_{d}$ | $\mathcal{N}(0, \sigma_d)$ | 行业随机效应 |

#### PyMC 实现

```python
import pymc as pm

with pm.Model() as vote_model:
    # ========== 超先验 ==========
    sigma_partner = pm.HalfNormal('sigma_partner', sigma=0.5)
    sigma_season = pm.HalfNormal('sigma_season', sigma=0.3)
    sigma_industry = pm.HalfNormal('sigma_industry', sigma=0.5)
    
    # ========== 固定效应 ==========
    beta_0 = pm.Normal('beta_0', mu=10, sigma=2)
    beta_score = pm.Normal('beta_score', mu=0, sigma=0.5)
    beta_age = pm.Normal('beta_age', mu=0, sigma=0.1)
    
    # ========== 随机效应 ==========
    alpha_partner = pm.Normal('alpha_partner', mu=0, sigma=sigma_partner,
                              shape=n_partners)
    gamma_season = pm.Normal('gamma_season', mu=0, sigma=sigma_season,
                             shape=n_seasons)
    delta_industry = pm.Normal('delta_industry', mu=0, sigma=sigma_industry,
                               shape=n_industries)
    
    # ========== 残差 ==========
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # ========== 线性预测器 ==========
    mu = (beta_0 + 
          beta_score * scores_normalized +
          beta_age * ages_normalized +
          alpha_partner[partner_idx] +
          gamma_season[season_idx] +
          delta_industry[industry_idx])
    
    # ========== 似然函数 ==========
    log_votes = pm.Normal('log_votes', mu=mu, sigma=sigma, shape=n_obs)
    
    # ========== MCMC采样 ==========
    trace = pm.sample(draws=2000, tune=1000, chains=2, 
                      random_seed=42, return_inferencedata=True)
```

#### 后验预测

```python
def sample_votes_posterior(contestants, n_samples=1000):
    """
    从后验分布采样投票
    """
    samples = np.zeros((n_samples, len(contestants)))
    
    # 从后验中随机抽取参数组合
    for s in range(n_samples):
        # 抽取参数
        beta_0 = posterior_samples['beta_0'][s]
        beta_score = posterior_samples['beta_score'][s]
        # ... 其他参数
        
        for i, row in contestants.iterrows():
            # 计算期望对数投票
            mu = (beta_0 + 
                  beta_score * normalize(row['total_score']) +
                  alpha_partner[row['partner_idx']] +
                  gamma_season[row['season_idx']])
            
            # 采样
            log_vote = np.random.normal(mu, sigma)
            samples[s, i] = np.exp(log_vote)
        
        # 归一化到总票数
        samples[s] = samples[s] / samples[s].sum() * total_votes
    
    return samples
```

#### MCMC 诊断

```python
# 收敛诊断
print(az.summary(trace, var_names=['beta_0', 'beta_score', 'sigma']))

# 检查项目:
# - R-hat ≈ 1.0 (< 1.01 为佳): 链间收敛
# - ESS > 400: 有效样本量
# - 无发散 (divergences = 0)
```

**推断方法**：PyMC + NUTS (No-U-Turn Sampler) MCMC

**优点**：
- 完整的不确定性量化
- 自动学习随机效应
- 可解释的层次结构

**缺点**：计算量较大

---

## 🔄 投票估计器整合

`VoteEstimator` 类整合三种模型，提供统一接口：

```python
class VoteEstimator:
    def __init__(self, model_type='ensemble'):
        """
        model_type: 'baseline', 'constrained', 'bayesian', 'ensemble'
        """
        self.baseline = BaselineModel()
        self.constrained = ConstrainedOptimizationModel()
        self.bayesian = BayesianVoteModel()
        
        # 集成权重
        self.weights = {'baseline': 0.2, 'constrained': 0.5, 'bayesian': 0.3}
    
    def estimate(self, season_week_data, elimination_info):
        """
        集成估计: 加权平均三种模型的结果
        """
        v_baseline = self.baseline.estimate_all_weeks(season_week_data)
        v_constrained = self.constrained.estimate_all_weeks(season_week_data, elimination_info)
        v_bayesian = self.bayesian.estimate_all_weeks(season_week_data)
        
        ensemble = {}
        for key in season_week_data.keys():
            ensemble[key] = (
                self.weights['baseline'] * v_baseline[key] +
                self.weights['constrained'] * v_constrained[key] +
                self.weights['bayesian'] * v_bayesian[key]
            )
        
        return ensemble
```

---

## 📈 模型评估指标

### 1. 一致性检验 (Consistency Check)

验证估计的投票是否能正确预测淘汰结果：

$$\text{Accuracy} = \frac{\text{正确预测的淘汰数}}{\text{总淘汰周次数}}$$

#### 实现逻辑

```python
def check_elimination_consistency(estimates, elimination_info):
    """
    检验每周淘汰预测的正确性
    """
    results = []
    
    for (season, week), est in estimates.items():
        # 获取实际淘汰者
        actual = elimination_info.query(f'season=={season} and week=={week}')
        if len(actual) == 0:
            continue
        actual_name = actual.iloc[0]['eliminated_name']
        
        # 预测淘汰者
        scores, votes = est['scores'], est['votes']
        
        if season in RANK_SEASONS:
            combined = compute_rank_combined_score(scores, votes)
            pred_idx = np.argmax(combined)  # 最高者被淘汰
        else:
            combined = compute_percent_combined_score(scores, votes)
            pred_idx = np.argmin(combined)  # 最低者被淘汰
        
        pred_name = est['names'][pred_idx]
        
        results.append({
            'season': season,
            'week': week,
            'actual': actual_name,
            'predicted': pred_name,
            'is_correct': actual_name == pred_name
        })
    
    return pd.DataFrame(results)
```

**底2准确率**：被淘汰选手是否在预测的最后两名中

### 2. 不确定性度量 (Uncertainty Measure)

#### 从后验样本计算

```python
def compute_uncertainty(samples):
    """
    samples: (n_samples, n_contestants)
    """
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    
    # 变异系数
    cv = std / mean
    
    # 95% 可信区间
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)
    ci_width = ci_upper - ci_lower
    
    # 确定性分类
    certainty = np.where(cv < 0.1, 'High',
                np.where(cv < 0.3, 'Medium', 'Low'))
    
    return {
        'mean': mean,
        'std': std,
        'cv': cv,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'certainty': certainty
    }
```

**变异系数 (CV)**：
$$CV = \frac{\sigma_V}{\mu_V}$$

**确定性分类**：
| CV 范围 | 确定性等级 |
|---------|-----------|
| CV < 0.1 | 高 (High) |
| 0.1 ≤ CV < 0.3 | 中 (Medium) |
| CV ≥ 0.3 | 低 (Low) |

### 3. 排名相关性

```python
from scipy import stats

def compute_rank_correlations(judge_scores, estimated_votes):
    """
    计算评委排名与投票排名的相关性
    """
    score_ranks = np.argsort(np.argsort(-judge_scores)) + 1
    vote_ranks = np.argsort(np.argsort(-estimated_votes)) + 1
    
    # Kendall's τ
    tau, p_tau = stats.kendalltau(score_ranks, vote_ranks)
    
    # Spearman's ρ  
    rho, p_rho = stats.spearmanr(score_ranks, vote_ranks)
    
    return {'kendall_tau': tau, 'spearman_rho': rho}
```

- **Kendall's τ**：评委排名与投票排名的相关性
- **Spearman's ρ**：等级相关系数

---

## 📁 输出文件说明

| 文件 | 说明 |
|------|------|
| `vote_estimates.csv` | 每位选手每周的投票估计值 |
| `consistency_results.csv` | 每周淘汰预测的正确性 |
| `incorrect_predictions.csv` | 预测错误的案例分析 |
| `figures/` | 可视化图表目录 |

### vote_estimates.csv 字段

| 字段 | 说明 |
|------|------|
| `season` | 赛季编号 |
| `week` | 周次 |
| `celebrity_name` | 选手姓名 |
| `judge_score` | 评委总分 |
| `estimated_votes` | 估计投票数 |

---

## 🎨 可视化输出

程序自动生成以下图表：

1. **投票估计柱状图**：展示各选手的估计投票与评委得分对比
2. **置信区间图**：带95%可信区间的投票估计
3. **一致性热力图**：各赛季各周的预测准确率
4. **不确定性分布图**：CV分布与确定性等级占比
5. **准确率汇总图**：总体与分赛季的预测准确率

---

## 🔬 争议案例分析

模型特别关注以下争议赛季：

| 赛季 | 争议选手 | 现象 |
|------|---------|------|
| S2 | Jerry Rice | 得分高但意外淘汰 |
| S4 | Billy Ray Cyrus | 得分低但进入决赛 |
| S11 | Bristol Palin | 得分垫底但获第三名 |
| S27 | Bobby Bones | 低分夺冠引发争议 |

这些案例反映了评委评分与观众投票的显著分歧，是验证模型的重要测试集。

---

## ⚙️ 配置参数

在 `config.py` 中可调整以下参数：

```python
# 模型配置
class ModelConfig:
    OPTIMIZATION_METHOD = 'SLSQP'    # 优化方法
    OPTIMIZATION_MAX_ITER = 1000     # 最大迭代次数
    MCMC_SAMPLES = 2000              # MCMC采样数
    MCMC_TUNE = 1000                 # 调优步数
    MCMC_CHAINS = 2                  # 马尔可夫链数
    RANDOM_SEED = 42                 # 随机种子
```

---

## 📚 参考文献

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.
2. Salvatier, J., et al. (2016). Probabilistic programming in Python using PyMC3.
3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

---

## 📝 许可证

本项目仅用于 2026 MCM 数学建模竞赛。

---

## 👥 贡献者

yanxiao 团队
