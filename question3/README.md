# Question 3: 影响因素分析

## 问题描述

利用包括观众投票估算结果在内的数据，开发模型分析专业舞者以及数据中可用的名人特征（年龄、行业等）的影响。

**核心问题：**
1. 这些因素对名人在比赛中的表现影响有多大？
2. 它们对评委得分和观众投票的影响是否相同？

## 项目结构

```
question3/
├── config.py           # 配置文件
├── data_loader.py      # 数据加载和预处理
├── models.py           # 分析模型（回归、随机森林、效应分析）
├── visualization.py    # 可视化模块
├── analysis.py         # 综合分析模块
├── main.py             # 主程序入口
├── requirements.txt    # 依赖包
├── vote_estimates.csv  # 预测的观众投票数据（输入）
└── outputs/            # 输出目录
    ├── figures/        # 可视化图表
    ├── tables/         # 数据表格
    └── reports/        # 分析报告
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行分析：
```bash
python main.py
```

## 分析方法

### 1. 效应分析 (Effect Analysis)
- 计算各因素（舞伴、年龄、行业、国籍）对评委得分和观众投票的效应
- 通过与总体均值对比量化效应大小

### 2. 回归分析 (Regression Analysis)
- OLS回归模型量化各因素的边际效应
- 混合效应模型考虑舞伴和赛季的随机效应

### 3. 随机森林分析 (Random Forest)
- 特征重要性分析
- 交叉验证评估模型性能
- SHAP值解释（可选）

### 4. 效应对比
- 对比各因素对评委得分和观众投票的不同影响
- 识别评委偏好 vs 观众偏好的差异

## 输出内容

### 图表
- `partner_effect_comparison.png` - 舞伴效应对比图
- `age_effect.png` - 年龄效应图
- `industry_effect.png` - 行业效应图
- `feature_importance_*.png` - 特征重要性图
- `feature_importance_comparison.png` - 特征重要性对比
- `effect_heatmap.png` - 效应热力图
- `analysis_dashboard.png` - 综合分析仪表板

### 表格
- `partner_effect.csv` - 舞伴效应数据
- `age_effect.csv` - 年龄效应数据
- `industry_effect.csv` - 行业效应数据
- `effect_comparison.csv` - 效应对比数据
- `*_feature_importance.csv` - 特征重要性数据

### 报告
- `analysis_report.md` - 分析报告

## 关键发现

1. **舞伴效应**：专业舞伴的选择显著影响选手表现，某些舞伴在提升技术分方面更强，另一些在吸引投票方面更有优势。

2. **行业效应**：运动员往往获得较高观众投票但评委得分较低；演员表现更均衡。

3. **年龄效应**：年轻选手更受观众欢迎，中年选手更受评委青睐。

4. **效应差异**：评委更注重技术表现，观众更看重知名度和亲和力。
