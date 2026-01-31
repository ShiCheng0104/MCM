# Question 2: 投票方法对比分析

## 问题描述

利用观众投票估算结果和其他数据，分析两种投票组合方法（排名法和百分比法）的差异及影响。

### 子问题

1. **方法对比**：比较和对比节目使用的两种投票组合方法在各季产生的结果
2. **争议案例分析**：考察两种方法在存在争议的特定选手身上的应用情况
3. **方法推荐**：基于分析结果，推荐未来赛季使用的方法

## 目录结构

```
question2/
├── analysis/                    # 分析模块
│   ├── __init__.py
│   ├── voting_methods.py       # 两种投票方法实现
│   ├── controversy_analysis.py # 争议案例分析
│   └── method_comparison.py    # 方法对比与推荐
├── outputs/                     # 输出目录
│   ├── figures/                # 图表
│   ├── method_comparison.csv   # 方法对比结果
│   ├── controversy_analysis.csv # 争议案例分析
│   └── recommendation_report.md # 推荐报告
├── config.py                   # 配置文件
├── main.py                     # 主程序
├── vote_estimates.csv          # 观众投票估计数据（输入）
└── README.md                   # 本文件
```

## 两种投票方法

### 排名法 (Rank Method)
- **使用赛季**：第1-2季，第28-34季
- **计算方式**：综合排名 = 评委得分排名 + 观众投票排名
- **淘汰规则**：综合排名最高者（数字最大）被淘汰

### 百分比法 (Percent Method)
- **使用赛季**：第3-27季
- **计算方式**：综合百分比 = 评委得分占比 + 观众投票占比
- **淘汰规则**：综合百分比最低者被淘汰

## 争议案例

| 选手 | 赛季 | 最终名次 | 争议描述 |
|------|------|----------|----------|
| Jerry Rice | S2 | 2nd | 5周评委最低分仍获亚军 |
| Billy Ray Cyrus | S4 | 5th | 6周评委最低分仍获第5名 |
| Bristol Palin | S11 | 3rd | 12次最低评委分仍获季军 |
| Bobby Bones | S27 | 1st | 持续低分仍获冠军 |

## 运行方式

```bash
cd question2
python main.py
```

## 输出说明

1. **method_comparison.csv**：各周次两种方法的淘汰结果对比
2. **controversy_analysis.csv**：争议案例的详细分析
3. **recommendation_report.md**：完整的分析报告和推荐建议
4. **figures/**：可视化图表
   - `method_comparison_overview.png`：方法对比概览
   - `controversy_trajectories.png`：争议选手轨迹图

## 主要发现

（运行程序后更新）

## 推荐结论

（运行程序后更新）
