# Google Trends 名人热度爬取工具

## 📋 简介

这个工具可以帮助你从Google Trends爬取400个名人在特定时间段的搜索热度数据。

## 🚀 安装依赖

```bash
pip install pytrends pandas requests
```

或者使用requirements文件：

```bash
pip install -r requirement.txt
```

## 📁 文件说明

- **google_trends_scraper.py** - 完整的爬取脚本（适合大批量数据）
- **quick_trends_example.py** - 快速示例脚本（适合测试和小批量）
- **2026_MCM_Problem_C_Data.csv** - 包含名人数据的CSV文件

## 💻 使用方法

### 方法1：运行完整爬取脚本（推荐）

爬取所有400+名人的数据：

```bash
python google_trends_scraper.py
```

该脚本会：
- 自动从CSV文件读取所有名人
- 查询每个名人的Google Trends数据
- 保存结果到两个文件：
  - `google_trends_results.json` - 完整的趋势数据
  - `google_trends_summary.csv` - 汇总数据（平均值、最大值、最小值）

### 方法2：使用快速示例脚本

先测试少量数据：

```bash
python quick_trends_example.py
```

## ⚙️ 自定义参数

### 修改时间范围

在脚本中找到 `timeframe` 参数，可以设置为：

```python
timeframe = 'today 1-m'   # 过去1个月
timeframe = 'today 3-m'   # 过去3个月
timeframe = 'today 12-m'  # 过去12个月（默认）
timeframe = 'today 5-y'   # 过去5年
timeframe = '2023-01-01 2024-01-01'  # 自定义日期范围
```

### 修改地理位置

```python
geo = 'US'  # 美国
geo = 'GB'  # 英国
geo = 'CN'  # 中国
geo = ''    # 全球
```

### 修改请求延迟

为避免被Google封禁，建议设置合理的延迟：

```python
scraper.scrape_all(
    timeframe='today 12-m',
    geo='US',
    delay_range=(2, 5),  # 每次请求间隔2-5秒
    batch_size=5         # 每5个请求增加额外延迟
)
```

## 📊 输出数据格式

### JSON文件（google_trends_results.json）

```json
[
  {
    "keyword": "Taylor Swift",
    "timeframe": "today 12-m",
    "geo": "US",
    "success": true,
    "average": 67.5,
    "max": 100,
    "min": 45,
    "data": {
      "2024-01-01": 65,
      "2024-01-08": 72,
      ...
    }
  }
]
```

### CSV文件（google_trends_summary.csv）

| celebrity_name | timeframe | geo | success | average_trend | max_trend | min_trend |
|----------------|-----------|-----|---------|---------------|-----------|-----------|
| Taylor Swift   | today 12-m| US  | True    | 67.5          | 100       | 45        |
| LeBron James   | today 12-m| US  | True    | 52.3          | 89        | 32        |

## ⚠️ 注意事项

1. **请求频率限制**
   - Google Trends对请求频率有限制
   - 建议每次请求间隔2-5秒
   - 大批量查询时（400个名人）建议分批进行

2. **可能的错误**
   - 429错误：请求过快，增加延迟时间
   - 连接超时：网络问题，检查网络连接
   - 数据为空：该名人搜索量太低或名字不匹配

3. **数据说明**
   - 趋势值范围：0-100（相对值，不是绝对搜索量）
   - 100表示在指定时间和地区的最高搜索热度
   - 0表示搜索量不足

## 🔧 高级用法

### 1. 分批爬取

如果担心一次性爬取太多数据，可以分批：

```python
# 修改 google_trends_scraper.py 中的 main() 函数
celebrities = scraper.load_celebrities()
batch1 = celebrities[0:100]   # 第1批
batch2 = celebrities[100:200] # 第2批
# ...以此类推
```

### 2. 断点续传

如果中途中断，可以检查已保存的数据，只爬取未完成的：

```python
import json
import pandas as pd

# 读取已有结果
try:
    with open('google_trends_results.json', 'r') as f:
        existing_results = json.load(f)
    completed = {r['keyword'] for r in existing_results}
except:
    completed = set()

# 只爬取未完成的
all_celebrities = scraper.load_celebrities()
remaining = [c for c in all_celebrities if c not in completed]
print(f"还需爬取 {len(remaining)} 个名人")
```

### 3. 比较多个时间段

```python
# 比较不同时间段的热度变化
timeframes = ['today 1-m', 'today 3-m', 'today 12-m']
for tf in timeframes:
    scraper.scrape_all(timeframe=tf, geo='US')
    scraper.save_results(
        output_file=f'results_{tf.replace(" ", "_")}.json',
        csv_output=f'summary_{tf.replace(" ", "_")}.csv'
    )
```

## 📈 数据分析示例

爬取完数据后，可以进行分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取汇总数据
df = pd.read_csv('google_trends_summary.csv')

# 找出最热门的前10个名人
top10 = df.nlargest(10, 'average_trend')
print(top10[['celebrity_name', 'average_trend']])

# 绘制柱状图
plt.figure(figsize=(12, 6))
plt.barh(top10['celebrity_name'], top10['average_trend'])
plt.xlabel('Average Trend Score')
plt.title('Top 10 Celebrities by Google Trends')
plt.tight_layout()
plt.savefig('top10_trends.png')
```

## 🐛 常见问题

**Q: 为什么有些名人的数据是0？**
A: 可能是该名人搜索量太低，或者名字拼写不准确。

**Q: 收到429错误怎么办？**
A: 增加 `delay_range` 的值，比如改为 `(5, 10)`。

**Q: 能否加快爬取速度？**
A: 不建议，Google Trends有严格的速率限制，太快会被封IP。

**Q: 数据不准确怎么办？**
A: Google Trends的数据是相对值和估算值，仅供参考。

## 📞 技术支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 网络连接是否正常
3. 延迟时间是否合理
4. CSV文件路径是否正确

## 📝 许可证

MIT License
