# Model Evaluation and Promotion
# 模型评估与推广

本模块提供对2026 MCM Problem C四个问题模型的全面评估和推广分析。

## 项目结构

```
model_evaluation/
├── config.py                 # 配置文件
├── main.py                   # 主程序入口
├── q1_evaluation.py          # Q1模型评估
├── q2_evaluation.py          # Q2模型评估
├── q3_evaluation.py          # Q3模型评估
├── q4_evaluation.py          # Q4模型评估
├── sensitivity_analysis.py   # 敏感性分析
├── promotion_analysis.py     # 推广分析
├── visualization.py          # 可视化模块
├── requirements.txt          # 依赖包
└── outputs/                  # 输出目录
    ├── figures/              # 可视化图表
    ├── tables/               # 数据表格
    └── reports/              # 分析报告
```

## 评估内容

### 1. 模型评估 (Model Evaluation)

#### Q1: 观众投票估计模型
- **交叉验证**: 5折交叉验证
- **时序验证**: 训练/测试季节分割
- **Bootstrap置信区间**: 95%置信区间
- **敏感性分析**: 参数扰动稳定性

#### Q2: 投票方法对比
- **方法一致性**: 两种方法结果对比
- **准确率分析**: 与历史结果匹配度
- **统计显著性**: McNemar检验
- **稳健性分析**: 跨时期表现

#### Q3: 影响因素分析
- **交叉验证R²**: 模型解释力
- **特征重要性稳定性**: Bootstrap分析
- **效应一致性**: 效应方向检验
- **模型对比**: RF vs 线性回归

#### Q4: 动态投票系统
- **综合性能**: Composite Score
- **Pareto效率**: 多目标优化
- **争议率优化**: 最优范围分析
- **排名稳定性**: 跨指标一致性

### 2. 敏感性分析 (Sensitivity Analysis)

- **参数敏感性**: 关键参数±5%, ±10%扰动
- **数据敏感性**: 数据噪声影响
- **特征敏感性**: 特征移除影响
- **阈值敏感性**: Q4系统阈值优化
- **蒙特卡洛分析**: 不确定性量化

### 3. 推广分析 (Promotion Analysis)

- **可迁移性**: 不同应用领域适用性
- **可扩展性**: 数据/计算/特征扩展能力
- **实际考虑**: 部署、维护、成本
- **推广建议**: 短期/中期/长期应用

## 使用方法

```bash
cd model_evaluation
python main.py
```

## 输出说明

### 图表
- `model_radar.png`: 模型综合评估雷达图
- `cv_comparison.png`: 交叉验证结果对比
- `temporal_validation.png`: 时序验证结果
- `dashboard.png`: 综合评估仪表板
- `promotion_analysis.png`: 推广分析图

### 表格
- `evaluation_summary.csv`: 评估摘要表

### 报告
- `evaluation_report.md`: 完整评估报告

## 关键指标

| Model | Primary Metric | Target | Status |
|-------|---------------|--------|--------|
| Q1 | Prediction Accuracy | ≥80% | ✓ |
| Q2 | Method Agreement | ≥70% | ✓ |
| Q3 | CV R² | ≥0.6 | ✓ |
| Q4 | Composite Score | ≥0.8 | ✓ |

## 评估标准

### 准确性 (Accuracy)
- 模型预测与实际结果的匹配程度

### 稳健性 (Robustness)
- 模型对参数变化和数据扰动的稳定性

### 泛化能力 (Generalization)
- 模型在新数据上的表现

### 创新性 (Innovation)
- 模型相对于基准的改进程度

## 依赖项

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```
