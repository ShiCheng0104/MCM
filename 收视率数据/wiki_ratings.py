import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def scrape_ratings(season_num, max_retries=3):
    """
    爬取某一季的评分数据
    :param season_num: 季数
    :param max_retries: 最大重试次数
    """
    url = f"https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_{season_num}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 重试逻辑
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Season {season_num}: HTTP {response.status_code} (尝试 {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避: 2s, 4s, 8s
                    continue
                return False
            break
        except Exception as e:
            print(f"Season {season_num}: 请求失败 - {e} (尝试 {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避: 2s, 4s, 8s
                continue
            return False
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 定义清理函数：删除括号中的引用标记
    def clean_data(text):
        return re.sub(r'\[\d+\]', '', text).strip()
    
    # 找到包含"Viewers"的表格
    ratings_table = None
    for table in soup.find_all("table", {"class": "wikitable"}):
        content = table.get_text(strip=True)
        if "Viewers" in content and ("million" in content or "%" in content):
            rows = table.find_all("tr")
            if rows and len(rows) > 2:
                ratings_table = table
                break
    
    if not ratings_table:
        print(f"Season {season_num}: 没有数据")
        return False
    
    # 提取表格的所有数据（包括表头）
    rows = ratings_table.find_all("tr")
    
    # 提取表头
    header_row = rows[0]
    headers = [col.get_text(strip=True) for col in header_row.find_all(["th", "td"])]
    
    # 提取数据行
    data = []
    for row in rows[1:]:  # 跳过表头
        cols = [clean_data(col.get_text(strip=True)) for col in row.find_all(["td", "th"])]
        # 如果列数与表头不匹配，填充空值或截断
        if len(cols) < len(headers):
            cols.extend([''] * (len(headers) - len(cols)))
        elif len(cols) > len(headers):
            cols = cols[:len(headers)]
        if cols:
            data.append(cols)
    
    if not data:
        print(f"Season {season_num}: 没有数据")
        return False
    
    # 保存为 CSV，包含列标题
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(f"ratings_{season_num}.csv", index=False, encoding="utf-8-sig")
    print(f"Season {season_num}: 已保存 ratings_{season_num}.csv (共 {len(data)} 行, {len(headers)} 列)")
    return True

if __name__ == "__main__":
    failed_seasons = []
    for season in range(1, 35):  # 1 到 34 季
        success = scrape_ratings(season, max_retries=3)
        if not success:
            failed_seasons.append(season)
    
    # 打印失败季数的汇总
    if failed_seasons:
        print(f"\n失败的季数: {failed_seasons}")
    else:
        print("\n✅ 所有季数爬取成功！")
