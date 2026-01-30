# Dancing with the Stars (美国版) 评分数据说明

## 📊 数据概况

本项目爬取了《Dancing with the Stars》美国版 1-34季的电视评分数据，来源为 Wikipedia。

### 数据范围

- **成功保存**：32个季度的CSV文件（Season 1-11, 13-34）
- **缺失数据**：Season 12、Season 31（无相关数据）

---

## 📝 数据格式说明

### 1. CSV文件结构

每个季度对应一个CSV文件，文件名为 `ratings_{season_num}.csv`，包含该季度所有集数的评分信息。

**例如：**

- `ratings_1.csv` - 第1季数据
- `ratings_2.csv` - 第2季数据
- `ratings_27.csv` - 第27季数据

### 2. 列说明

各季度的表格结构略有不同，常见列包括：

| 列名 | 说明 | 示例 |
|-----|------|------|
| No. | 集数编号 | 1, 2, 3, Special |
| Title | 节目标题 | "Episode 101", "Dance-Off" |
| Air date | 首播日期 | "June 1, 2005" |
| Timeslot (ET) | 播出时间(东部时间) | "Wednesday 9:00 p.m." |
| Rating/share | 收视率和份额 | "4.3/12" |
| Viewers(millions) | 观看人数(百万) | "13.48" |

---

## ⚠️ 数据使用注意事项

### 1. **周统计处理**

**需要自行处理：** 表格按集数列出，不是按周汇总的。如需按周统计，需要：

- 通过 `Air date` 列解析日期
- 根据日期判断属于第几周
- 对同一周的集数进行汇总统计（求和或平均）

**示例逻辑：**

```python
# 伪代码
for index, row in df.iterrows():
    date = parse_date(row['Air date'])
    week = get_week_number(date)
    # 按week分组统计
```

### 2. **Special 节目**

部分季度包含特殊节目，如：

- "Dance-Off"
- "Dance-Off Results Show"
- "Finale"

**处理建议：**

- 如果需要统计常规节目，应排除 No. 为 "Special" 或包含特殊标记的行
- 这些节目通常在 No. 列会明确标注为 "Special"

### 3. **Viewers 单位说明**

**默认单位：millions（百万）**

- 列名中会明确注明：`Viewers(millions)` 或 `Viewers (in millions)`
- 数值表示观看人数的百万倍
- 例如：`13.48` 表示 1348 万人

### 4. **数据错位问题**

⚠️ **重要：** 部分季度存在表格列错位的情况

**示例（Season 4）：**

```
No.,Title,Air date,Timeslot (ET),Viewers(millions)
1,"""Episode 401""","March 19, 2007",Monday 8:00 p.m.,21.80
2,"""Episode 402""","March 26, 2007",20.42,
```

注意第2行缺少 "Timeslot" 列，导致 Viewers 的值 (20.42) 出现在 Timeslot 列的位置。

**处理建议：**

- 手工审查数据，特别是数值列
- 根据上下文推断正确的列值
- 实际使用时应进行数据验证和清洗

### 5. **缺失数据**

| 季度 | 状态 | 原因 |
|-----|------|------|
| Season 12 | ❌ 缺失 | Wikipedia 上无相关评分表 |
| Season 31 | ❌ 缺失 | Wikipedia 上无相关评分表 |

---

## 📋 数据质量说明

### ✅ 已处理项

- ✓ 删除了 Wikipedia 的引用标记 `[1]`, `[2]` 等
- ✓ 保留了原始表格的所有列
- ✓ 自动处理了列数不匹配的情况（填充空值）

### ⚠️ 已知问题

- 某些季度表格列数不一致（如 Season 4）
- 某些列可能包含合并单元格导致的错位
- 引号和特殊字符保持原样（如 `"""Episode 101"""` 中的多个引号）

### 💡 建议做法

1. 使用前先检查 CSV 文件，了解其结构
2. 对数据类型进行必要的转换和清洗
3. 对异常值进行手工验证
4. 建立数据验证流程确保分析的准确性

---

## 🔍 数据样本

### Season 1 (完整表格 - 6列)

```
No.,Title,Air date,Timeslot (ET),Rating/share(18–49),Viewers(millions)
1,"""Episode 101""","June 1, 2005",Wednesday 9:00 p.m.,4.3/12,13.48
2,"""Episode 102""","June 8, 2005",,4.8/14,15.09
3,"""Episode 103""","June 15, 2005",,4.8/14,15.67
Special,"""Dance-Off""","September 20, 2005",Tuesday 8:30 p.m.,2.8/7,10.91
```

### Season 11 (简化表格 - 2列)

```
Week,Viewers (in millions)
PerformanceShow,ResultsShow
1,21
2,21.341
3,19.889
```

---

## 📚 使用建议

### 数据读取

```python
import pandas as pd

# 读取数据
df = pd.read_csv('ratings_1.csv')

# 查看结构
print(df.head())
print(df.columns)
print(df.dtypes)
```

### 基本数据清洗

```python
# 排除 Special 节目
regular_episodes = df[df['No.'] != 'Special']

# 转换数值列
df['Viewers'] = pd.to_numeric(df['Viewers(millions)'], errors='coerce')
```

---

## 📞 数据来源

- **源网站**：Wikipedia - Dancing with the Stars (American TV series)
- **URL模式**：`https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_{n}`
- **爬虫工具**：Python (requests, BeautifulSoup, pandas)
- **爬取日期**：2026年1月30日

---

## � 文件清单

```
ratings_1.csv   - Season 1 (8行, 6列)
ratings_2.csv   - Season 2 (16行, 6列)
ratings_3.csv   - Season 3 (20行, 6列)
ratings_4.csv   - Season 4 (19行, 5列)
ratings_5.csv   - Season 5 (21行, 5列)
ratings_6.csv   - Season 6 (20行, 6列)
ratings_7.csv   - Season 7 (21行, 6列)
ratings_8.csv   - Season 8 (21行, 6列)
ratings_9.csv   - Season 9 (21行, 6列)
ratings_10.csv  - Season 10 (19行, 6列)
ratings_11.csv  - Season 11 (11行, 2列)
ratings_12.csv  - Season 12 ❌ 缺失
ratings_13.csv  - Season 13 (23行, 4列)
ratings_14.csv  - Season 14 (19行, 7列)
ratings_15.csv  - Season 15 (19行, 7列)
ratings_16.csv  - Season 16 (20行, 7列)
ratings_17.csv  - Season 17 (12行, 7列)
ratings_18.csv  - Season 18 (12行, 6列)
ratings_19.csv  - Season 19 (15行, 6列)
ratings_20.csv  - Season 20 (14行, 7列)
ratings_21.csv  - Season 21 (14行, 7列)
ratings_22.csv  - Season 22 (11行, 7列)
ratings_23.csv  - Season 23 (15行, 4列)
ratings_24.csv  - Season 24 (11行, 5列)
ratings_25.csv  - Season 25 (12行, 5列)
ratings_26.csv  - Season 26 (4行, 5列)
ratings_27.csv  - Season 27 (11行, 6列)
ratings_28.csv  - Season 28 (11行, 5列)
ratings_29.csv  - Season 29 (11行, 5列)
ratings_30.csv  - Season 30 (11行, 9列)
ratings_31.csv  - Season 31 ❌ 缺失
ratings_32.csv  - Season 32 (11行, 6列)
ratings_33.csv  - Season 33 (10行, 10列)
ratings_34.csv  - Season 34 (12行, 10列)
```

**统计信息：**

- 总计：**32个CSV文件**
- 总数据行数：**475行**
- 缺失季数：Season 12, Season 31

---

**最后更新**：2026年1月30日
