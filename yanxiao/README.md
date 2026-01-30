# 问题1：观众投票估计模型

## 项目结构

```
yanxiao/
├── README.md                    # 项目说明
├── main.py                      # 主程序入口
├── config.py                    # 配置参数
├── data/
│   └── __init__.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # 数据预处理模块
│   ├── vote_estimator.py       # 投票估计核心模型
│   ├── consistency_check.py    # 一致性检验模块
│   ├── uncertainty_measure.py  # 确定性度量模块
│   └── utils.py                # 工具函数
├── models/
│   ├── __init__.py
│   ├── constrained_optimization.py  # 约束优化模型
│   ├── bayesian_model.py            # 贝叶斯层次模型
│   └── baseline_model.py            # 基线模型
├── visualization/
│   ├── __init__.py
│   └── plots.py                # 可视化函数
└── outputs/                    # 输出结果目录
```

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main.py
```

## 模型说明

### 1. 约束优化模型
基于淘汰结果约束，反推满足条件的观众投票分布

### 2. 贝叶斯层次模型
使用层次结构建模选手、舞伴、赛季等随机效应

### 3. 基线模型
基于评委得分的简单比例估计
