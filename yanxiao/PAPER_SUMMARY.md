# 问题1建模总结 - 供论文使用

## 核心公式

### 贝叶斯层级模型

$$
\log(V_{ijk}) = \beta_0 + \beta_1 \cdot S_{ijk} + \beta_2 \cdot A_i + \alpha_{p(i)} + \gamma_k + \delta_{d(i)} + \epsilon_{ijk}
$$

其中：
- $V_{ijk}$: 选手 $i$ 在第 $k$ 季第 $j$ 周的投票数
- $S_{ijk}$: 评委评分
- $A_i$: 选手年龄
- $\alpha_{p(i)}$: 专业舞伴随机效应
- $\gamma_k$: 赛季随机效应  
- $\delta_{d(i)}$: 选手行业随机效应
- $\epsilon_{ijk} \sim N(0, \sigma^2)$: 残差

### 先验分布

$$
\begin{aligned}
\beta_0 &\sim N(10, 25) \\
\beta_1, \beta_2 &\sim N(0, 1) \\
\alpha_p &\sim N(0, \sigma_p^2), \quad \sigma_p \sim \text{Half-}N(0, 1) \\
\gamma_k &\sim N(0, \sigma_\gamma^2), \quad \sigma_\gamma \sim \text{Half-}N(0, 1) \\
\delta_d &\sim N(0, \sigma_\delta^2), \quad \sigma_\delta \sim \text{Half-}N(0, 1) \\
\sigma &\sim \text{Half-}N(0, 1)
\end{aligned}
$$

### 约束优化模型

对于每个赛季-周次组合 $(k, j)$，求解：

$$
\min_{\mathbf{V}} \sum_{i=1}^{n} \left( V_i - \hat{V}_i \right)^2
$$

约束条件：
$$
\begin{aligned}
& V_i \geq 0, \quad \forall i \\
& \sum_{i=1}^{n} V_i = V_{total} \\
& V_{elim} \leq V_i, \quad \forall i \neq elim \quad \text{(淘汰者投票最低)}
\end{aligned}
$$

---

## 核心结果表

### 表1: 模型参数估计

| Parameter | Mean | SD | 95% HDI | Interpretation |
|-----------|------|-----|---------|----------------|
| $\beta_0$ | 8.664 | 1.858 | [5.479, 12.025] | Baseline log-votes |
| $\beta_1$ | 0.049 | 0.554 | [-0.993, 0.921] | Score effect |
| $\beta_2$ | -0.001 | 0.095 | [-0.196, 0.161] | Age effect |
| $\sigma$ | 1.854 | 0.571 | [1.094, 2.977] | Residual SD |

### 表2: 预测准确性

| Metric | Value |
|--------|-------|
| Elimination Accuracy | 70.45% |
| Bottom-2 Accuracy | 92.05% |
| Kendall's τ | 0.838 |
| Spearman's ρ | 0.848 |

### 表3: 按投票方法的准确性

| Voting Method | Seasons | Accuracy | Bottom-2 |
|--------------|---------|----------|----------|
| Percentage-based | 3-27 | 80.30% | 100% |
| Rank-based | 1-2, 28-34 | 40.91% | 68.18% |

---

## 关键图表建议

### Figure 1: 模型框架图
```
[数据] → [预处理] → [贝叶斯模型] → [约束优化] → [投票估计]
                          ↓                ↓
                    [后验分布]      [一致性验证]
```

### Figure 2: 预测准确性热力图
- X轴: 周次 (1-11)
- Y轴: 赛季 (1-34)
- 颜色: 预测正确(绿)/底2命中(黄)/错误(红)

### Figure 3: 投票估计分布
- 展示典型周次的投票估计条形图
- 附带95%置信区间

### Figure 4: 两种投票方法对比
- 双面柱状图对比准确率差异
- 强调rank方法的挑战

---

## 论文语言模板

### 方法描述

> We developed a Bayesian hierarchical model to estimate viewer votes for each contestant in each week of competition. The model accounts for judge scores, contestant demographics, and incorporates random effects for professional partners, seasons, and contestant industries. Posterior inference was conducted using Markov Chain Monte Carlo (MCMC) with the No-U-Turn Sampler (NUTS).

### 结果描述

> Our model achieved an elimination prediction accuracy of 70.45% and a bottom-2 accuracy of 92.05% across 264 elimination weeks. The model performance varied significantly between voting methods: percentage-based voting seasons (S3-27) yielded 80.30% accuracy, while rank-based seasons (S1-2, S28-34) achieved only 40.91%. This disparity reflects the inherent information loss in rank-based reporting and the influence of fan mobilization effects that override dance performance.

### 局限性描述

> Our model has several limitations. First, the Bayesian inference showed 93 divergences during MCMC sampling, indicating potential issues with posterior geometry. Second, the model cannot capture "fan effects" where contestants with strong social media followings or political significance receive votes disproportionate to their dance ability (e.g., Bristol Palin in Season 11, Sean Spicer in Season 28). Third, uncertainty quantification relies on a fixed coefficient of variation, which may underestimate true uncertainty for edge cases.

---

## 核心发现总结

1. **评委评分是投票的显著预测因子** ($\beta_1 > 0$)
2. **年龄对投票无显著影响** ($\beta_2 \approx 0$)
3. **投票方法显著影响可预测性** (80% vs 41%)
4. **"粉丝效应"是主要误差来源**
5. **模型整体排序能力强** (τ = 0.84)

---

*供论文撰写使用*
