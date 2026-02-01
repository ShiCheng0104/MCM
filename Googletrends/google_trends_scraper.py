#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends爬取脚本 - 按season分组比较名人热度
同一season的名人一起查询，使用统一的归一化基准
"""

import pandas as pd
import time
import random
import itertools
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import json

class GoogleTrendsScraper:
    def __init__(self, csv_file='2026_MCM_Problem_C_Data.csv', season_file='季度日期表.csv'):
        """初始化爬虫
        
        Args:
            csv_file: 包含名人名字的CSV文件路径
            season_file: 包含season时间表的CSV文件路径
        """
        self.csv_file = csv_file
        self.season_file = season_file
        self.pytrends = None
        self.results = []
        self.season_dates = None
        self.request_count = 0  # 请求计数器
        self.last_request_time = 0  # 上次请求时间
        self.consecutive_failures = 0  # 连续失败次数
    
    def generate_name_variants(self, name):
        """生成名字的多种变体用于搜索
        
        Args:
            name: 原始名人名字
            
        Returns:
            list: 包含所有变体的列表（去重）
        """
        variants = set()
        variants.add(name)  # 原始名字
        
        # 1. 生成大小写组合
        # 只对包含字母的名字处理
        if any(c.isalpha() for c in name):
            # 全小写
            variants.add(name.lower())
            # 全大写
            variants.add(name.upper())
            # 首字母大写，其余小写
            variants.add(name.capitalize())
            # 标题格式（每个单词首字母大写）
            variants.add(name.title())
            
            # 对于短名字（<=4个字符），生成所有大小写组合
            if len(name) <= 4 and name.replace(" ", "").replace("'", "").isalpha():
                clean_name = name.replace(" ", "").replace("'", "")
                if 2 <= len(clean_name) <= 3:
                    # 生成所有可能的大小写组合
                    import itertools
                    for combo in itertools.product(*[(c.lower(), c.upper()) for c in clean_name]):
                        variant = ''.join(combo)
                        variants.add(variant)
        
        # 2. 如果名字中包含撇号，为所有变体生成无撇号版本
        if "'" in name:
            # 复制当前所有变体
            variants_with_apostrophe = list(variants)
            # 为每个变体都生成去除撇号的版本
            for variant in variants_with_apostrophe:
                if "'" in variant:
                    variants.add(variant.replace("'", ""))
        
        # 移除空字符串
        variants.discard("")
        
        result = sorted(list(variants))
        if len(result) > 1:
            print(f"    生成 {len(result)} 个变体: {', '.join(result)}")
        
        return result
        
    def init_pytrends(self):
        """初始化pytrends对象"""
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            print("✓ Pytrends初始化成功")
            return True
        except Exception as e:
            print(f"✗ Pytrends初始化失败: {e}")
            return False
    
    def load_season_dates(self):
        """加载season时间表"""
        try:
            df = pd.read_csv(self.season_file, encoding='utf-8')
            season_dict = {}
            for _, row in df.iterrows():
                season = int(row['季度'])
                date_range = row['日期范围']
                dates = date_range.split(' - ')
                start_date = datetime.strptime(dates[0].strip(), "%B %d, %Y")
                end_date = datetime.strptime(dates[1].strip(), "%B %d, %Y")
                season_dict[season] = {
                    'start': start_date,
                    'end': end_date,
                    'timeframe': f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
                }
            self.season_dates = season_dict
            print(f"✓ 成功加载 {len(season_dict)} 个season的时间信息")
            return True
        except Exception as e:
            print(f"✗ 加载season时间表失败: {e}")
            return False
    
    def load_celebrities_by_season(self):
        """从CSV文件中加载名人，按season分组"""
        try:
            df = pd.read_csv(self.csv_file)
            
            # 按season分组
            season_groups = {}
            for _, row in df.iterrows():
                season = int(row['season'])
                if season not in season_groups:
                    season_groups[season] = []
                
                season_groups[season].append({
                    'name': row['celebrity_name'],
                    'industry': row['celebrity_industry'],
                    'placement': row['placement']
                })
            
            print(f"✓ 成功加载 {len(df)} 条名人参赛记录，分布在 {len(season_groups)} 个season")
            return season_groups
        except Exception as e:
            print(f"✗ 加载CSV文件失败: {e}")
            return {}
    
    def get_trend_data_batch(self, keywords, timeframe, geo='US', max_retries=3):
        """获取一批关键词的趋势数据（最多5个）
        
        Args:
            keywords: 关键词列表（最多5个）
            timeframe: 时间范围
            geo: 地理位置
            max_retries: 最大重试次数（仅用于非限流错误）
        
        Returns:
            dict: 每个关键词的趋势数据
        """
        # Google Trends最多支持5个关键词
        if len(keywords) > 5:
            keywords = keywords[:5]
        
        # 智能延迟：根据请求次数和失败情况动态调整
        base_delay = 10  # 基础延迟从10秒起步
        
        # 如果有连续失败，增加延迟
        if self.consecutive_failures > 0:
            base_delay += self.consecutive_failures * 10  # 每次失败增加10秒
            print(f"  ⚠️ 检测到 {self.consecutive_failures} 次连续失败，增加延迟")
        
        # 每10次请求后增加延迟
        if self.request_count > 0 and self.request_count % 10 == 0:
            base_delay += 20  # 每10次请求额外增加20秒
            print(f"  ⚠️ 已发送 {self.request_count} 次请求，增加延迟避免限流")
        
        # 确保两次请求间隔至少base_delay秒
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < base_delay:
                wait = base_delay - elapsed + random.uniform(2, 5)
                print(f"  ⏳ 智能延迟 {wait:.1f} 秒（基础延迟={base_delay}秒）...")
                time.sleep(wait)
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        attempt = 0
        rate_limit_retry_count = 0  # 限流重试计数（无上限）
        
        while True:
            try:
                # 构建payload
                self.pytrends.build_payload(
                    kw_list=keywords,
                    timeframe=timeframe,
                    geo=geo
                )
                
                # 获取interest over time
                interest_over_time = self.pytrends.interest_over_time()
                
                # 成功请求，重置连续失败计数
                self.consecutive_failures = 0
                
                results = {}
                
                if not interest_over_time.empty:
                    # 移除isPartial列
                    if 'isPartial' in interest_over_time.columns:
                        interest_over_time = interest_over_time.drop(columns=['isPartial'])
                    
                    # 处理每个关键词
                    for keyword in keywords:
                        if keyword in interest_over_time.columns:
                            trend_values = interest_over_time[keyword]
                            trend_dict = {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date): int(value) 
                                          for date, value in trend_values.items()}
                            
                            results[keyword] = {
                                'data': trend_dict,
                                'sum': int(trend_values.sum()),
                                'average': float(trend_values.mean()),
                                'max': int(trend_values.max()),
                                'min': int(trend_values.min()),
                                'count': len(trend_values),
                                'success': True
                            }
                        else:
                            results[keyword] = {
                                'data': None,
                                'sum': 0,
                                'average': 0,
                                'max': 0,
                                'min': 0,
                                'count': 0,
                                'success': False
                            }
                else:
                    for keyword in keywords:
                        results[keyword] = {
                            'data': None,
                            'sum': 0,
                            'average': 0,
                            'max': 0,
                            'min': 0,
                            'count': 0,
                            'success': False
                        }
                
                return results
                
            except Exception as e:
                error_msg = str(e)
                
                # 增加连续失败计数
                self.consecutive_failures += 1
                
                # 如果是429错误，无限重试，等待时间递增
                if '429' in error_msg or 'rate limit' in error_msg.lower() or 'too many requests' in error_msg.lower():
                    rate_limit_retry_count += 1
                    # 等待时间：60秒起步，每次递增60秒，最多600秒（10分钟）
                    wait_time = min(rate_limit_retry_count * 60, 600)
                    
                    print(f"  🚫 遇到限流(429)，第 {rate_limit_retry_count} 次重试（无上限）")
                    print(f"  ⏰ 等待 {wait_time} 秒（{wait_time/60:.1f} 分钟）冷却...")
                    print(f"  📊 当前已发送 {self.request_count} 次请求，连续失败 {self.consecutive_failures} 次")
                    
                    time.sleep(wait_time)
                    
                    # 每5次限流后重新初始化pytrends（更换session）
                    if rate_limit_retry_count % 5 == 0:
                        print(f"  🔄 尝试重新初始化连接...")
                        self.init_pytrends()
                    
                    continue  # 继续重试，不增加attempt
                else:
                    # 其他错误，使用有限重试
                    attempt += 1
                    if attempt < max_retries:
                        wait_time = 30  # 增加到30秒
                        print(f"  ⚠️ 遇到错误: {e}，第 {attempt} 次重试，等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ✗ 达到最大重试次数({max_retries})，放弃: {e}")
                        # 返回失败结果
                        return {keyword: {
                            'error': str(e),
                            'sum': 0,
                            'average': 0,
                            'max': 0,
                            'min': 0,
                            'count': 0,
                            'success': False
                        } for keyword in keywords}
    
    def get_celebrity_combined_trend(self, celebrity_name, timeframe, geo='US', baseline_keyword=None, baseline_value=None):
        """获取一个名人所有变体的综合热度数据
        
        Args:
            celebrity_name: 名人原始名字
            timeframe: 时间范围
            geo: 地理位置
            baseline_keyword: 固定基准词（用于归一化）
            baseline_value: 基准词的初始平均值
            
        Returns:
            dict: 包含综合热度数据
        """
        # 生成所有变体
        variants = self.generate_name_variants(celebrity_name)
        
        # 分批查询，每批最多4个变体 + 1个基准词 = 5个
        batch_size = 4
        batch_results = {}
        
        for i in range(0, len(variants), batch_size):
            batch = variants[i:i+batch_size]
            
            # 如果有基准词，添加到查询中
            if baseline_keyword:
                batch = [baseline_keyword] + batch
            
            results = self.get_trend_data_batch(batch, timeframe, geo)
            
            # 如果有基准词和基准值，进行归一化
            if baseline_keyword and baseline_value and baseline_keyword in results:
                current_baseline = results[baseline_keyword].get('average', 0)
                if current_baseline > 0:
                    ratio = baseline_value / current_baseline
                    print(f"      归一化比例: {ratio:.4f} (基准词在本批={current_baseline:.2f})")
                    # 归一化当前批次的数据
                    for variant in variants:
                        if variant in results and results[variant].get('success'):
                            results[variant]['average'] *= ratio
                            results[variant]['sum'] = int(results[variant]['sum'] * ratio)
            
            # 只保存变体的结果，不保存基准词的结果
            for variant in variants:
                if variant in results:
                    batch_results[variant] = results[variant]
        
        # 合并所有变体的结果
        combined_average = 0
        combined_sum = 0
        combined_max = 0
        success_count = 0
        
        for variant in variants:
            if variant in batch_results and batch_results[variant].get('success'):
                combined_average += batch_results[variant]['average']
                combined_sum += batch_results[variant]['sum']
                combined_max = max(combined_max, batch_results[variant]['max'])
                success_count += 1
        
        if success_count > 0:
            print(f"    ✓ 成功查询 {success_count}/{len(variants)} 个变体，综合平均热度={combined_average:.2f}")
            return {
                'average': combined_average,
                'sum': combined_sum,
                'max': combined_max,
                'variants_searched': len(variants),
                'variants_success': success_count,
                'success': True
            }
        else:
            print(f"    ✗ 所有变体查询失败")
            return {
                'average': 0,
                'sum': 0,
                'max': 0,
                'variants_searched': len(variants),
                'variants_success': 0,
                'success': False
            }
    
    def scrape_season_with_normalization(self, season, celebrities, season_info, geo='US'):
        """爬取一个season的所有名人数据（使用固定基准词归一化）
        
        Args:
            season: season编号
            celebrities: 该season的名人列表
            season_info: season时间信息
            geo: 地理位置
        """
        timeframe = season_info['timeframe']
        
        # 使用固定的基准词（热度适中且稳定）
        # RuPaul是节目主持人，在整个时间段内都有稳定的热度
        baseline_keyword = "RuPaul"
        baseline_value = None
        
        print(f"\n  🎯 固定基准词: {baseline_keyword}")
        print(f"  所有名人将通过基准词进行归一化，确保可比性")
        
        # 首先查询基准词，获取基准值
        print(f"\n  [0/{len(celebrities)}] 查询基准词: {baseline_keyword}")
        baseline_results = self.get_trend_data_batch([baseline_keyword], timeframe, geo)
        
        if baseline_keyword in baseline_results and baseline_results[baseline_keyword].get('success'):
            baseline_value = baseline_results[baseline_keyword]['average']
            print(f"    📌 基准词平均热度: {baseline_value:.2f}")
        else:
            print(f"    ⚠️ 基准词查询失败，将无法进行归一化")
            baseline_value = None
        
        # 逐个查询每个名人（包含所有变体）
        for idx, celeb in enumerate(celebrities, 1):
            print(f"\n  [{idx}/{len(celebrities)}] 查询: {celeb['name']}")
            
            # 使用基准词归一化
            result = self.get_celebrity_combined_trend(
                celeb['name'], 
                timeframe, 
                geo,
                baseline_keyword=baseline_keyword,
                baseline_value=baseline_value
            )
            
            # 保存结果
            if result['success']:
                self.results.append({
                    'name': celeb['name'],
                    'season': season,
                    'average': result['average'],
                    'variants_searched': result['variants_searched'],
                    'success': True
                })
            else:
                # 失败的也记录，便于追踪
                print(f"    ⚠️ {celeb['name']} 查询失败，跳过")
            
            # 每个名人之间延迟
            if idx < len(celebrities):
                # 动态延迟：根据失败情况调整
                base_delay = 15 if self.consecutive_failures == 0 else 30
                delay = random.uniform(base_delay, base_delay + 10)
                print(f"  ⏳ 等待 {delay:.1f} 秒...")
                time.sleep(delay)
    
    def scrape_all(self, geo='US', delay_range=(5, 10)):
        """按season分组爬取所有名人的趋势数据
        
        Args:
            geo: 地理位置
            delay_range: 每个season之间的延迟范围（秒）
        """
        # 加载season时间表
        if not self.load_season_dates():
            return
        
        # 加载名人数据（按season分组）
        season_groups = self.load_celebrities_by_season()
        if not season_groups:
            print("没有找到名人数据")
            return
        
        if not self.init_pytrends():
            return
        
        print(f"\n开始按season分组爬取Google Trends数据...")
        print(f"地理位置: {geo if geo else '全球'}")
        print(f"注意: 每个名人会搜索多个变体（移除撇号、大小写组合），结果加和")
        print(f"关键: 所有查询使用同一基准人物归一化，确保数据可比性")
        print("=" * 70)
        
        total_seasons = len(season_groups)
        
        for season_idx, (season, celebrities) in enumerate(sorted(season_groups.items()), 1):
            # 获取对应season的时间范围
            if season not in self.season_dates:
                print(f"\n[Season {season}] ✗ 未找到时间信息，跳过")
                continue
            
            season_info = self.season_dates[season]
            
            print(f"\n{'='*70}")
            print(f"Season {season} ({season_idx}/{total_seasons})")
            print(f"时间: {season_info['start'].strftime('%Y-%m-%d')} 到 {season_info['end'].strftime('%Y-%m-%d')}")
            print(f"参赛者: {len(celebrities)} 人")
            print(f"{'='*70}")
            
            # 爬取该season的数据
            self.scrape_season_with_normalization(season, celebrities, season_info, geo)
            
            # 每个season完成后立即保存结果
            print(f"\n💾 保存Season {season}的数据...")
            self.save_results(
                output_file='google_trends_results.json',
                csv_output='google_trends_summary.csv',
                append=True
            )
            print(f"✓ 已保存 {len(self.results)} 条数据")
            
            # 每个season处理完后延迟
            if season_idx < total_seasons:
                delay = random.uniform(*delay_range)
                print(f"\n⏳ Season {season} 完成，等待 {delay:.1f} 秒...")
                time.sleep(delay)
        
        print("\n" + "=" * 70)
        print(f"✓ 爬取完成！共 {len(self.results)} 条数据")
        
    def standardize_results(self):
        """对结果进行Min-Max归一化（按season分组）
        
        Returns:
            list: 包含归一化结果的列表
        """
        if not self.results:
            print("没有数据可标准化")
            return []
        
        # 提取所有成功的结果
        valid_results = [r for r in self.results if r.get('success')]
        if not valid_results:
            print("没有有效数据可标准化")
            return []
        
        # 按season分组
        season_groups = {}
        for result in valid_results:
            season = result.get('season', 0)
            if season not in season_groups:
                season_groups[season] = []
            season_groups[season].append(result)
        
        # 对每个season独立进行Min-Max归一化
        standardized = []
        
        print(f"\n标准化统计（按季度）:")
        for season in sorted(season_groups.keys()):
            season_data = season_groups[season]
            averages = [r['average'] for r in season_data]
            min_val = min(averages)
            max_val = max(averages)
            
            print(f"  Season {season}:")
            print(f"    - 原始平均值范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"    - 参赛者数量: {len(season_data)}")
            
            # 归一化该season的数据
            for result in season_data:
                if max_val > min_val:
                    normalized = (result['average'] - min_val) / (max_val - min_val)
                else:
                    # 如果所有值相同，设为0.5
                    normalized = 0.5
                
                standardized.append({
                    'celebrity_name': result['name'],
                    'season': season,
                    'normalized_average': round(normalized, 6),
                    'variants_searched': result.get('variants_searched', 1)
                })
        
        print(f"\n  - 总处理数据: {len(standardized)} 条")
        print(f"  - 归一化范围: [0, 1] (每个season独立)")
        
        return standardized
    
    def save_results(self, output_file='google_trends_results.json', csv_output='google_trends_summary.csv', append=False):
        """保存结果到文件
        
        Args:
            output_file: JSON输出文件
            csv_output: CSV输出文件（包含归一化结果）
            append: 是否追加模式（用于增量保存）
        """
        if not self.results:
            print("没有数据可保存")
            return
        
        # 保存原始JSON数据
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
            if not append:
                print(f"✓ 原始数据已保存到: {output_file}")
        except Exception as e:
            print(f"✗ 保存JSON失败: {e}")
        
        # 进行Min-Max归一化
        standardized_results = self.standardize_results()
        
        # 保存归一化CSV
        if standardized_results:
            try:
                df = pd.DataFrame(standardized_results)
                # 按season和归一化值排序
                df = df.sort_values(['season', 'normalized_average'], ascending=[True, False])
                df.to_csv(csv_output, index=False, encoding='utf-8-sig')
                print(f"✓ 归一化数据已保存到: {csv_output}")
                print(f"\n数据概览（按季度分组）:")
                
                # 按season分组显示
                for season in sorted(df['season'].unique()):
                    season_df = df[df['season'] == season]
                    print(f"\n  Season {season} ({len(season_df)} 人):")
                    # 显示名字、归一化值和变体数量
                    for _, row in season_df.iterrows():
                        print(f"    {row['celebrity_name']:30s} {row['normalized_average']:.6f} ({row['variants_searched']} 变体)")
                    
            except Exception as e:
                print(f"✗ 保存CSV失败: {e}")

def main():
    """主函数"""
    print("=" * 70)
    print("Google Trends 名人热度爬取工具 - 按Season分组比较")
    print("=" * 70)
    
    # 创建爬虫实例
    scraper = GoogleTrendsScraper(
        csv_file='2026_MCM_Problem_C_Data.csv',
        season_file='季度日期表.csv'
    )
    
    geo = 'US'
    
    # 开始爬取
    scraper.scrape_all(
        geo=geo,
        delay_range=(15, 25)  # 每个season间隔15-25秒，避免被限流
    )
    
    # 保存最终结果（带完整统计信息）
    print("\n" + "=" * 70)
    print("生成最终统计报告...")
    print("=" * 70)
    scraper.save_results(
        output_file='google_trends_results.json',
        csv_output='google_trends_summary.csv',
        append=False
    )
    
if __name__ == '__main__':
    main()
