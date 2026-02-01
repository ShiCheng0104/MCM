"""
精确投票反推模型

严格按照题目规则：
1. 排名法 (赛季1-2, 28-34): 评委排名 + 观众排名 = 综合排名，最高者淘汰
2. 百分比法 (赛季3-27): 评委百分比 + 观众百分比 = 综合百分比，最低者淘汰

核心思路：
- 已知：评委得分、谁被淘汰
- 求解：观众投票（使被淘汰者的综合得分最低/排名最高）

扩展特征：
- 观众人数：反映可参与投票的人数基数
- Google Trends：反映名人热度，按季度归一化（0-1）
"""
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution, LinearConstraint
from scipy.stats import rankdata
import warnings

# 赛季分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))  # 排名法赛季
SEASONS_PERCENT_METHOD = list(range(3, 28))  # 百分比法赛季

# 导入基线模型用于无淘汰周次的预测
try:
    from .baseline_model import BaselineModel
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False
    print("Warning: baseline_model not available, will use simple estimation for non-elimination weeks")


class PreciseVoteModel:
    """
    精确投票反推模型
    
    关键：根据淘汰结果反推出满足约束的观众投票
    
    扩展特征：
    - 观众人数 (viewers): 影响投票基数，观众多则投票总量大
    - 名人热度 (popularity): Google Trends归一化值，热度高的名人更易获得投票
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 基线模型用于无淘汰周次预测
        if BASELINE_AVAILABLE:
            self.baseline_model = BaselineModel(alpha=1.2, noise_level=0.15)
        else:
            self.baseline_model = None
        
        # 学习到的投票偏好因子
        self.partner_effects = {}
        self.industry_effects = {}
        
        # 全局参数
        self.base_vote_share = 0.5  # 基础投票份额
        self.score_influence = 0.3  # 评分对投票的影响
        self.popularity_influence = 0.25  # 热度对投票的影响权重
        self.viewers_influence = 0.15  # 观众人数对投票的影响权重
        
        # 标准化参数
        self.age_mean = 0
        self.age_std = 1
        self.viewers_mean = 0
        self.viewers_std = 1
        
        # 外部数据缓存
        self.google_trends_data = None  # {(celebrity, season): normalized_average}
        self.viewership_data = None  # {season: {week: viewers}}
        
        # 结果存储
        self.results_df = None
        self.week_results = {}
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """训练模型"""
        print("正在训练精确投票反推模型...")
        
        # 0. 加载外部数据（Google Trends + 观众人数）
        self._load_google_trends_data()
        self._load_viewership_data()
        
        # 0.5 将外部数据合并到weekly_data
        self._merge_external_data(weekly_data)
        
        # 1. 学习投票偏好（舞伴/行业效应 + 热度/观众效应）
        self._learn_vote_preferences(weekly_data, elimination_info)
        
        # 2. 对每个周次反推投票
        self._solve_all_weeks(weekly_data, elimination_info)
        
        # 注: 准确率验证在 predict_elimination() 中单独进行
        
        self.is_fitted = True
        print("模型训练完成!")
    
    def _load_google_trends_data(self):
        """加载Google Trends数据"""
        print("  正在加载Google Trends数据...")
        
        # 尝试多个可能的路径
        possible_paths = [
            'MCM/Googletrends/google_trends_summary.csv',
            'Googletrends/google_trends_summary.csv',
            '../Googletrends/google_trends_summary.csv',
            'MCM/google_trends_summary.csv',
        ]
        
        trends_df = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    trends_df = pd.read_csv(path)
                    print(f"    从 {path} 加载了 {len(trends_df)} 条Google Trends记录")
                    break
                except Exception as e:
                    print(f"    警告: 读取 {path} 失败: {e}")
        
        if trends_df is None:
            print("    警告: 无法加载Google Trends数据，将使用默认值0")
            self.google_trends_data = {}
            return
        
        # 构建查找字典 {(celebrity_name, season): normalized_average}
        self.google_trends_data = {}
        for _, row in trends_df.iterrows():
            key = (row['celebrity_name'], row['season'])
            self.google_trends_data[key] = row['normalized_average']
        
        print(f"    成功加载 {len(self.google_trends_data)} 条名人热度数据")
    
    def _load_viewership_data(self):
        """加载观众人数数据"""
        print("  正在加载观众人数数据...")
        
        # 尝试多个可能的路径
        possible_dirs = [
            'MCM/收视率数据/processed',
            '收视率数据/processed',
            '../收视率数据/processed',
        ]
        
        viewership_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                viewership_dir = d
                break
        
        if viewership_dir is None:
            print("    警告: 无法找到观众人数数据目录，将使用默认值")
            self.viewership_data = {}
            return
        
        self.viewership_data = {}  # {season: {week: viewers}}
        
        for season in range(1, 35):
            # 优先读取merged文件，否则读取普通文件
            merged_file = os.path.join(viewership_dir, f'processed_ratings_{season}_merged.csv')
            normal_file = os.path.join(viewership_dir, f'processed_ratings_{season}.csv')
            
            file_to_read = merged_file if os.path.exists(merged_file) else normal_file
            
            if os.path.exists(file_to_read):
                try:
                    df = pd.read_csv(file_to_read, usecols=['week', 'viewers'])
                    self.viewership_data[season] = {}
                    
                    for _, row in df.iterrows():
                        week = row['week']
                        viewers_str = str(row['viewers'])
                        
                        # 处理逗号分割的多次播出数据（merged文件）
                        if ',' in viewers_str:
                            # 取多次播出的总和（投票可能来自多次播出的观众）
                            viewer_values = [float(v.strip()) for v in viewers_str.split(',')]
                            total_viewers = np.sum(viewer_values)
                        else:
                            total_viewers = float(viewers_str)
                        
                        self.viewership_data[season][week] = total_viewers
                        
                except Exception as e:
                    print(f"    警告: 读取Season {season}观众数据失败: {e}")
        
        # 补充缺失的第12和31季数据（用相邻季度平均值）
        for missing_season in [12, 31]:
            if missing_season not in self.viewership_data:
                print(f"    补充Season {missing_season}缺失数据（使用相邻季度平均值）")
                
                prev_season = missing_season - 1
                next_season = missing_season + 1
                
                if prev_season in self.viewership_data and next_season in self.viewership_data:
                    self.viewership_data[missing_season] = {}
                    
                    all_weeks = set(self.viewership_data[prev_season].keys()) | \
                               set(self.viewership_data[next_season].keys())
                    
                    for week in all_weeks:
                        prev_val = self.viewership_data[prev_season].get(week, None)
                        next_val = self.viewership_data[next_season].get(week, None)
                        
                        if prev_val is not None and next_val is not None:
                            self.viewership_data[missing_season][week] = (prev_val + next_val) / 2
                        elif prev_val is not None:
                            self.viewership_data[missing_season][week] = prev_val
                        elif next_val is not None:
                            self.viewership_data[missing_season][week] = next_val
        
        print(f"    成功加载 {len(self.viewership_data)} 个季度的观众人数数据")
    
    def _merge_external_data(self, weekly_data: pd.DataFrame):
        """将外部数据合并到weekly_data"""
        print("  正在合并外部数据...")
        
        # 确定名人列名
        name_col = 'celebrity_name' if 'celebrity_name' in weekly_data.columns else 'celebrity'
        
        # 合并Google Trends数据（popularity）
        popularity_list = []
        for _, row in weekly_data.iterrows():
            celeb = row[name_col]
            season = row['season']
            key = (celeb, season)
            
            if key in self.google_trends_data:
                popularity_list.append(self.google_trends_data[key])
            else:
                popularity_list.append(0.0)  # 缺失则为0
        
        weekly_data['popularity'] = popularity_list
        
        # 用季度内中位数填充缺失值
        for season in weekly_data['season'].unique():
            season_mask = weekly_data['season'] == season
            season_median = weekly_data.loc[season_mask, 'popularity'].median()
            if pd.notna(season_median) and season_median > 0:
                missing_mask = season_mask & (weekly_data['popularity'] == 0)
                weekly_data.loc[missing_mask, 'popularity'] = season_median
        
        pop_coverage = (weekly_data['popularity'] > 0).mean()
        print(f"    Google Trends数据覆盖率: {pop_coverage:.1%}")
        
        # 合并观众人数数据（viewers）
        viewers_list = []
        for _, row in weekly_data.iterrows():
            season = row['season']
            week = row['week']
            
            if season in self.viewership_data and week in self.viewership_data[season]:
                viewers_list.append(self.viewership_data[season][week])
            else:
                viewers_list.append(np.nan)
        
        weekly_data['viewers'] = viewers_list
        
        # 填充缺失值（用全局中位数）
        median_viewers = weekly_data['viewers'].median()
        weekly_data['viewers'] = weekly_data['viewers'].fillna(median_viewers)
        
        # 标准化观众人数
        self.viewers_mean = weekly_data['viewers'].mean()
        self.viewers_std = weekly_data['viewers'].std()
        if self.viewers_std == 0:
            self.viewers_std = 1
        
        viewers_coverage = weekly_data['viewers'].notna().mean()
        print(f"    观众人数数据覆盖率: {viewers_coverage:.1%}")
        print(f"    观众人数范围: {weekly_data['viewers'].min():.1f} - {weekly_data['viewers'].max():.1f} 百万")
        
    def _learn_vote_preferences(self, weekly_data: pd.DataFrame, 
                                elimination_info: pd.DataFrame):
        """
        学习投票偏好：哪些舞伴/行业能获得更多投票支持
        
        方法：统计各舞伴/行业的"超预期存活率"
        
        扩展：考虑名人热度和观众人数的影响
        """
        print("  学习投票偏好...")
        
        # 标准化年龄
        ages = weekly_data['celebrity_age'].dropna()
        self.age_mean = ages.mean()
        self.age_std = ages.std() if ages.std() > 0 else 1
        
        partner_stats = {}  # {partner: [survive_boost, ...]}
        industry_stats = {}
        popularity_boosts = []  # 热度对存活的影响
        viewers_effects = []  # 观众人数对存活的影响
        
        # 确定名人列名
        name_col = 'celebrity_name' if 'celebrity_name' in weekly_data.columns else 'celebrity'
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = set(elim['eliminated_name'].tolist())
            n = len(group)
            
            # 计算评分排名
            scores = group['total_score'].values
            score_ranks = n - rankdata(scores, method='ordinal') + 1  # 1=最高分
            
            # 获取该周的观众人数（用于分析观众人数与投票的关系）
            week_viewers = group['viewers'].mean() if 'viewers' in group.columns else 0
            
            for idx, (_, row) in enumerate(group.iterrows()):
                name = row[name_col]
                partner = row.get('ballroom_partner', 'Unknown')
                industry = row.get('celebrity_industry', 'Unknown')
                score_rank = score_ranks[idx]
                popularity = row.get('popularity', 0)
                
                # 超预期存活：评分低但没被淘汰
                # 负向超预期：评分高但被淘汰
                if name in eliminated_names:
                    # 被淘汰：计算"提前淘汰程度"
                    boost = -(n - score_rank) / n  # 评分越高，被淘汰越意外
                    # 热度高的人被淘汰更意外
                    popularity_boosts.append((popularity, -1))  # 淘汰
                else:
                    # 存活：计算"超预期存活程度"
                    boost = (score_rank - 1) / n  # 评分越低，存活越意外
                    # 热度高的人存活是预期的
                    popularity_boosts.append((popularity, 1))  # 存活
                
                # 记录
                if partner not in partner_stats:
                    partner_stats[partner] = []
                partner_stats[partner].append(boost)
                
                if industry not in industry_stats:
                    industry_stats[industry] = []
                industry_stats[industry].append(boost)
        
        # 计算平均效应
        overall_mean = 0
        for partner, boosts in partner_stats.items():
            self.partner_effects[partner] = np.mean(boosts) - overall_mean
        
        for industry, boosts in industry_stats.items():
            self.industry_effects[industry] = np.mean(boosts) - overall_mean
        
        # 学习热度对存活的影响（正相关说明热度高更易存活）
        if popularity_boosts:
            high_pop = [outcome for (pop, outcome) in popularity_boosts if pop > 0.5]
            low_pop = [outcome for (pop, outcome) in popularity_boosts if pop <= 0.5]
            
            high_survival_rate = np.mean([1 if o > 0 else 0 for o in high_pop]) if high_pop else 0.5
            low_survival_rate = np.mean([1 if o > 0 else 0 for o in low_pop]) if low_pop else 0.5
            
            self.popularity_effect = high_survival_rate - low_survival_rate
            print(f"    热度效应: 高热度存活率={high_survival_rate:.1%}, 低热度存活率={low_survival_rate:.1%}")
            print(f"    热度提升存活率: {self.popularity_effect:+.1%}")
        else:
            self.popularity_effect = 0
        
        # 打印效应
        sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    舞伴效应 (Top 5, 正值=更多投票支持):")
        for p, e in sorted_partners[:5]:
            print(f"      {p}: {e:+.3f}")
        
        sorted_industries = sorted(self.industry_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    行业效应 (Top 5):")
        for ind, e in sorted_industries[:5]:
            print(f"      {ind}: {e:+.3f}")
    
    def _solve_all_weeks(self, weekly_data: pd.DataFrame, 
                         elimination_info: pd.DataFrame):
        """
        对每个周次反推投票
        
        扩展：考虑观众人数对投票总量的影响，以及名人热度对投票份额的影响
        """
        print("  反推各周次投票...")
        
        results = []
        success_count = 0
        total_count = 0        
        failed_weeks = []  # 记录失败的周次
        
        # 确定名人列名
        name_col = 'celebrity_name' if 'celebrity_name' in weekly_data.columns else 'celebrity'
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            # 获取该周的观众人数（用于计算投票总量）
            week_viewers = group['viewers'].mean() if 'viewers' in group.columns else 20.0
            # 基于观众人数估算投票总量（假设投票率约5%，观众单位为百万）
            estimated_total_votes = week_viewers * 1_000_000 * 0.05  # 百万观众 * 5%投票率
            estimated_total_votes = max(estimated_total_votes, 500_000)  # 至少50万票
            
            if len(elim) == 0:
                # 无淘汰周次：使用基线模型 + 先验偏好 + 热度预测投票
                method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
                
                # 准备选手数据
                contestants_no_elim = []
                for _, row in group.iterrows():
                    partner = row.get('ballroom_partner', 'Unknown')
                    industry = row.get('celebrity_industry', 'Unknown')
                    popularity = row.get('popularity', 0)
                    
                    prior_boost = (
                        self.partner_effects.get(partner, 0) +
                        self.industry_effects.get(industry, 0)
                    )
                    contestants_no_elim.append({
                        'name': row[name_col],
                        'score': row['total_score'],
                        'prior_boost': prior_boost,
                        'popularity': popularity
                    })
                
                scores = np.array([c['score'] for c in contestants_no_elim])
                prior_boosts = np.array([c['prior_boost'] for c in contestants_no_elim])
                popularities = np.array([c['popularity'] for c in contestants_no_elim])
                
                # 方法1：基于评分的幂次关系（alpha=1.2）
                if self.baseline_model is not None:
                    base_votes = self.baseline_model.estimate_votes(scores, total_votes=1_000_000)
                    base_shares = base_votes / 1_000_000
                else:
                    base_votes_raw = np.power(scores, 1.2)
                    base_shares = base_votes_raw / np.sum(base_votes_raw)
                
                # 方法2：结合先验偏好
                score_based = scores / np.sum(scores)
                prior_adjusted = score_based * (1 + prior_boosts * 0.3)
                prior_adjusted = prior_adjusted / np.sum(prior_adjusted)
                
                # 方法3：结合热度调整（热度高获得更多投票）
                popularity_factor = 1 + popularities * self.popularity_influence
                popularity_adjusted = base_shares * popularity_factor
                popularity_adjusted = popularity_adjusted / np.sum(popularity_adjusted)
                
                # 综合：基线50% + 先验20% + 热度30%
                vote_shares_est = (0.5 * base_shares + 
                                  0.2 * prior_adjusted + 
                                  0.3 * popularity_adjusted)
                vote_shares_est = vote_shares_est / np.sum(vote_shares_est)
                
                for i, c in enumerate(contestants_no_elim):
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity': c['name'],
                        'total_score': c['score'],
                        'estimated_votes': int(round(vote_shares_est[i] * estimated_total_votes)),
                        'vote_share': vote_shares_est[i],
                        'method': method,
                        'is_eliminated': False,
                        'popularity': c['popularity'],
                        'viewers': week_viewers
                    })
                continue
            
            eliminated_names = set(elim['eliminated_name'].tolist())
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            
            # 准备数据
            contestants = []
            for _, row in group.iterrows():
                partner = row.get('ballroom_partner', 'Unknown')
                industry = row.get('celebrity_industry', 'Unknown')
                popularity = row.get('popularity', 0)
                
                # 先验投票偏好（结合舞伴/行业效应 + 热度效应）
                prior_boost = (
                    self.partner_effects.get(partner, 0) +
                    self.industry_effects.get(industry, 0) +
                    popularity * self.popularity_influence  # 热度越高，投票越多
                )
                
                contestants.append({
                    'name': row[name_col],
                    'score': row['total_score'],
                    'is_eliminated': row[name_col] in eliminated_names,
                    'prior_boost': prior_boost,
                    'popularity': popularity,
                    'partner': partner,
                    'industry': industry
                })
            
            # 检查淘汰者是否在选手中
            valid_eliminated = [c for c in contestants if c['is_eliminated']]
            if not valid_eliminated:
                # 无淘汰周次：使用多种方法预测投票
                scores = np.array([c['score'] for c in contestants])
                prior_boosts = np.array([c['prior_boost'] for c in contestants])
                popularities = np.array([c['popularity'] for c in contestants])
                
                # 方法1：基于评分的幂次关系
                if self.baseline_model is not None:
                    base_votes = self.baseline_model.estimate_votes(scores, total_votes=1_000_000)
                    base_shares = base_votes / 1_000_000
                else:
                    base_votes_raw = np.power(scores, 1.2)
                    base_shares = base_votes_raw / np.sum(base_votes_raw)
                
                # 方法2：结合先验偏好调整
                score_based = scores / np.sum(scores)
                prior_adjusted = score_based * (1 + prior_boosts * 0.3)
                prior_adjusted = prior_adjusted / np.sum(prior_adjusted)
                
                # 方法3：热度调整
                popularity_factor = 1 + popularities * self.popularity_influence
                popularity_adjusted = base_shares * popularity_factor
                popularity_adjusted = popularity_adjusted / np.sum(popularity_adjusted)
                
                # 综合
                vote_shares_est = 0.5 * base_shares + 0.2 * prior_adjusted + 0.3 * popularity_adjusted
                vote_shares_est = vote_shares_est / np.sum(vote_shares_est)
                
                for i, c in enumerate(contestants):
                    results.append({
                        'season': season,
                        'week': week,
                        'celebrity': c['name'],
                        'total_score': c['score'],
                        'estimated_votes': int(round(vote_shares_est[i] * estimated_total_votes)),
                        'vote_share': vote_shares_est[i],
                        'method': method,
                        'is_eliminated': False,
                        'popularity': c['popularity'],
                        'viewers': week_viewers
                    })
                continue
            
            # 反推投票（传入热度信息）
            vote_shares, success = self._solve_votes_for_week(contestants, method)
            
            if success:
                success_count += 1
            else:
                failed_weeks.append((season, week))
                # 优化失败：使用备用方法估计（基于评分、先验和热度）
                scores_arr = np.array([c['score'] for c in contestants])
                prior_boosts = np.array([c['prior_boost'] for c in contestants])
                popularities = np.array([c['popularity'] for c in contestants])
                is_eliminated = np.array([c['is_eliminated'] for c in contestants])
                
                # 备用估计：评分比例 × (1 + 先验) × (1 + 热度)
                score_based = scores_arr / np.sum(scores_arr)
                popularity_factor = 1 + popularities * self.popularity_influence
                vote_shares = score_based * (1 + prior_boosts * 0.3) * popularity_factor
                vote_shares = vote_shares / np.sum(vote_shares)
                
                # 排名法兜底：确保备用估计也满足淘汰约束
                if method == 'rank':
                    elim_indices = np.where(is_eliminated)[0]
                    surv_indices = np.where(~is_eliminated)[0]
                    if len(elim_indices) > 0:
                        # 验证是否满足约束
                        if not self._verify_elimination(scores_arr, vote_shares, is_eliminated, method):
                            # 使用兜底方法强制构造
                            vote_shares = self._force_construct_rank_votes(
                                scores_arr, vote_shares, is_eliminated, elim_indices, surv_indices
                            )
                            # 再次验证
                            if self._verify_elimination(scores_arr, vote_shares, is_eliminated, method):
                                success = True
                                success_count += 1
                                failed_weeks.pop()  # 移除刚添加的失败记录
            
            # 所有周次都存储到week_results（包括失败的）
            self.week_results[(season, week)] = {
                'contestants': contestants,
                'vote_shares': vote_shares,
                'method': method,
                'success': success,
                'viewers': week_viewers
            }
                
            total_count += 1
            
            for i, c in enumerate(contestants):
                votes = int(round(vote_shares[i] * estimated_total_votes)) if vote_shares is not None else np.nan
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': c['name'],
                    'total_score': c['score'],
                    'estimated_votes': votes,
                    'vote_share': vote_shares[i] if vote_shares is not None else np.nan,
                    'method': method,
                    'is_eliminated': c['is_eliminated'],
                    'optimization_success': success,
                    'popularity': c['popularity'],
                    'viewers': week_viewers
                })
        
        print(f"    反推成功: {success_count}/{total_count} 周次")
        if failed_weeks:
            print(f"    优化失败的周次 ({len(failed_weeks)}): {failed_weeks[:5]}{'...' if len(failed_weeks) > 5 else ''}")
        
        self.results_df = pd.DataFrame(results)
        
        # 统计有效估计数
        valid_estimates = self.results_df.groupby(['season', 'week']).first()
        has_valid_votes = valid_estimates['estimated_votes'].notna().sum()
        print(f"    有效投票估计: {has_valid_votes} 个周次（包括优化失败但给出估计的）")
    
    def _solve_votes_for_week(self, contestants: List[Dict], method: str) -> Tuple[np.ndarray, bool]:
        """
        为单周反推投票份额
        
        约束：被淘汰者的综合得分必须是最低的（排名法）或最低的（百分比法）
        
        扩展：将名人热度纳入先验估计
        """
        n = len(contestants)
        scores = np.array([c['score'] for c in contestants])
        prior_boosts = np.array([c['prior_boost'] for c in contestants])
        popularities = np.array([c.get('popularity', 0) for c in contestants])
        is_eliminated = np.array([c['is_eliminated'] for c in contestants])
        
        # 被淘汰者索引
        elim_indices = np.where(is_eliminated)[0]
        surv_indices = np.where(~is_eliminated)[0]
        
        if len(elim_indices) == 0:
            return None, False
        
        def objective(vote_shares):
            """
            目标：使投票份额接近先验（基于评分+热度），同时满足淘汰约束
            
            扩展：先验中考虑热度因素
            """
            # 基于评分和热度的先验估计
            score_based = scores / np.sum(scores)
            
            # 热度调整：热度高的人预期获得更多投票
            popularity_factor = 1 + popularities * self.popularity_influence
            
            # 先验投票份额 = 评分比例 × (1 + 舞伴/行业效应) × 热度因子
            prior_votes = score_based * (1 + prior_boosts) * popularity_factor
            prior_votes = prior_votes / np.sum(prior_votes)
            
            # L2正则化损失
            reg_loss = np.sum((vote_shares - prior_votes) ** 2)
            
            # 淘汰约束违反惩罚
            if method == 'rank':
                # 排名法：使用可微分的软排名近似
                score_softrank = np.zeros(n)
                vote_softrank = np.zeros(n)
                temperature = 0.1  # 温度参数，越小越接近真实排名
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            score_softrank[i] += 1 / (1 + np.exp((scores[j] - scores[i]) / temperature))
                            vote_softrank[i] += 1 / (1 + np.exp((vote_shares[j] - vote_shares[i]) / temperature))
                
                # 综合软排名（越大 = 越差）
                combined_softrank = (n - score_softrank) + (n - vote_softrank)
                
                elim_combined = combined_softrank[elim_indices]
                surv_combined = combined_softrank[surv_indices]
                
                # 惩罚：确保所有被淘汰者的综合排名都大于所有存活者
                constraint_violation = 0
                for ec in elim_combined:
                    for sc in surv_combined:
                        if ec <= sc:
                            constraint_violation += (sc - ec + 0.5) ** 2
            else:
                combined = self._compute_combined_percent(scores, vote_shares)
                elim_combined = combined[elim_indices]
                surv_combined = combined[surv_indices]
                
                # 惩罚：如果有存活者的combined_percent <= 被淘汰者
                constraint_violation = 0
                for ec in elim_combined:
                    for sc in surv_combined:
                        if sc <= ec:
                            constraint_violation += (ec - sc + 0.01) ** 2
            
            return reg_loss + 1000 * constraint_violation
        
        # 初始值：基于评分、先验偏好和热度
        score_based = scores / np.sum(scores)
        popularity_factor = 1 + popularities * self.popularity_influence
        x0 = score_based * (1 + prior_boosts * 0.5) * popularity_factor
        x0 = x0 / np.sum(x0)

        # 边界：每个份额在[lb, ub]之间
        eps = 1e-6
        lb = 1e-3
        ub = 1.0 - 1e-3
        bounds = [(lb, ub)] * n

        # 确保初始值严格在边界内部
        x0 = np.clip(x0, lb + eps, ub - eps)
        # 再次归一化确保和为1
        x0 = x0 / np.sum(x0)
        # 最终裁剪确保边界
        x0 = np.clip(x0, lb, ub)
        
        # 约束：份额之和为1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # 优化（抑制边界警告）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Values in x were outside bounds')
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8}
            )
        
        vote_shares = result.x
        vote_shares = vote_shares / np.sum(vote_shares)  # 确保和为1
        
        # 验证约束是否满足
        success = self._verify_elimination(scores, vote_shares, is_eliminated, method)
        
        # 排名法兜底：如果优化后仍不满足约束，直接构造投票使被淘汰者综合排名最大
        if not success and method == 'rank':
            vote_shares = self._force_construct_rank_votes(
                scores, vote_shares, is_eliminated, elim_indices, surv_indices
            )
            success = self._verify_elimination(scores, vote_shares, is_eliminated, method)
        
        return vote_shares, success
    
    def _force_construct_rank_votes(self, scores: np.ndarray, vote_shares: np.ndarray,
                                     is_eliminated: np.ndarray,
                                     elim_indices: np.ndarray, surv_indices: np.ndarray) -> np.ndarray:
        """
        排名法兜底方法：直接构造观众投票数，确保被淘汰者综合排名最大
        
        核心思想：
        - 综合排名 = 评委排名 + 观众排名
        - 要使被淘汰者综合排名最大，需要让其观众排名也尽可能大（差）
        - 关键：被淘汰者必须获得最后n_elim个观众排名位置
        
        保证方法：
        - 存活者的投票份额全部 > 被淘汰者的投票份额
        - 这样被淘汰者的观众排名一定是 n_surv+1, n_surv+2, ..., n
        """
        n = len(scores)
        score_ranks = rankdata(-scores, method='ordinal')  # 最高分=1
        
        n_elim = len(elim_indices)
        n_surv = len(surv_indices)
        
        if n_surv == 0:
            # 特殊情况：所有人都被淘汰
            return vote_shares
        
        # 创建新的投票份额数组
        new_shares = np.zeros(n)
        
        # ============ 核心策略 ============
        # 1. 存活者份额范围: [0.1, 1.0] 按评分比例分配
        # 2. 被淘汰者份额范围: [0.001, 0.01] 确保全部低于存活者
        
        # 存活者份额
        surv_scores = scores[surv_indices]
        surv_scores_normalized = surv_scores / np.sum(surv_scores)
        # 映射到 [0.1, 0.9] 范围
        surv_min, surv_max = 0.1, 0.9
        surv_shares = surv_min + surv_scores_normalized * (surv_max - surv_min) * n_surv
        new_shares[surv_indices] = surv_shares
        
        # 被淘汰者份额：必须全部低于存活者最低份额
        min_surv_share = np.min(new_shares[surv_indices])
        
        # 被淘汰者按评委排名排序（评委分高的需要更低的投票来补偿）
        # 评委排名小 = 评委分高 = 需要更低的投票份额
        elim_with_ranks = [(idx, score_ranks[idx]) for idx in elim_indices]
        elim_sorted = sorted(elim_with_ranks, key=lambda x: x[1])  # 按评委排名排序
        
        # 分配被淘汰者份额：评委排名越好（小），投票份额越低
        elim_max = min_surv_share * 0.1  # 被淘汰者最高份额 = 存活者最低的10%
        elim_min = elim_max * 0.001  # 被淘汰者最低份额
        
        for i, (elim_idx, _) in enumerate(elim_sorted):
            # 按顺序递增分配（排名最好的获得最少）
            if n_elim > 1:
                ratio = i / (n_elim - 1)
            else:
                ratio = 0.5
            new_shares[elim_idx] = elim_min + ratio * (elim_max - elim_min)
        
        # 归一化
        new_shares = new_shares / np.sum(new_shares)
        
        # ============ 验证并强制修正 ============
        vote_ranks = rankdata(-new_shares, method='ordinal')
        combined_ranks = score_ranks + vote_ranks
        
        elim_min_combined = np.min(combined_ranks[elim_indices])
        surv_max_combined = np.max(combined_ranks[surv_indices])
        
        if elim_min_combined <= surv_max_combined:
            # 仍不满足，使用终极兜底：直接指定观众排名
            # 被淘汰者获得最后n_elim个排名位置
            
            # 为了让被淘汰者获得最后n_elim个观众排名，
            # 需要确保：所有被淘汰者份额 < 所有存活者份额
            
            # 重新分配：存活者份额递增，被淘汰者份额极小且递减
            new_shares = np.zeros(n)
            
            # 存活者：按评分排序分配递增份额
            surv_with_scores = [(idx, scores[idx]) for idx in surv_indices]
            surv_sorted = sorted(surv_with_scores, key=lambda x: x[1])
            
            for i, (surv_idx, _) in enumerate(surv_sorted):
                new_shares[surv_idx] = 1.0 + i * 0.1  # 1.0, 1.1, 1.2, ...
            
            # 被淘汰者：按评委排名排序，评委分越高份额越低
            for i, (elim_idx, _) in enumerate(elim_sorted):
                new_shares[elim_idx] = 0.001 * (0.1 ** i)  # 极小值递减
            
            # 归一化
            new_shares = new_shares / np.sum(new_shares)
        
        return new_shares
    
    def _compute_combined_rank(self, scores: np.ndarray, vote_shares: np.ndarray, n: int) -> np.ndarray:
        """
        计算排名法的综合排名
        
        规则：综合排名 = 评委排名 + 观众排名
        
        排名解释：
        - 排名1 = 第一名（最好）
        - 排名越大 = 名次越差
        - 综合排名数值最大者被淘汰（排名最低/最差）
        
        示例：
        选手A: 评分排名3 + 投票排名2 = 综合排名5
        选手B: 评分排名1 + 投票排名5 = 综合排名6 ← 被淘汰（数值最大）
        选手C: 评分排名2 + 投票排名1 = 综合排名3
        """
        # 评委排名：分数越高排名越好（数值越小）
        score_ranks = rankdata(-scores, method='ordinal')  # 最高分=1, 最低分=n
        
        # 观众排名：投票越多排名越好（数值越小）
        vote_ranks = rankdata(-vote_shares, method='ordinal')  # 最高票=1, 最低票=n
        
        # 综合排名（数值越大 = 排名越差 = 越可能被淘汰）
        combined_rank = score_ranks + vote_ranks
        
        return combined_rank
    
    def _compute_combined_percent(self, scores: np.ndarray, vote_shares: np.ndarray) -> np.ndarray:
        """
        计算百分比法的综合百分比
        
        综合百分比 = 评委百分比 + 观众百分比（越小越可能被淘汰）
        """
        # 评委百分比
        score_percent = scores / np.sum(scores)
        
        # 观众百分比 = vote_shares（已经是百分比）
        
        # 综合百分比
        combined_percent = score_percent + vote_shares
        
        return combined_percent
    
    def _verify_elimination(self, scores: np.ndarray, vote_shares: np.ndarray, 
                           is_eliminated: np.ndarray, method: str) -> bool:
        """验证淘汰约束是否满足"""
        n = len(scores)
        
        if method == 'rank':
            combined = self._compute_combined_rank(scores, vote_shares, n)
            # 被淘汰者的综合排名应该是最大的（数值最大 = 排名最差）
            # 处理多人淘汰：确保所有被淘汰者的排名都大于所有存活者
            elim_min = np.min(combined[is_eliminated])  # 被淘汰者中最好的
            surv_max = np.max(combined[~is_eliminated]) if np.any(~is_eliminated) else 0  # 存活者中最差的
            return elim_min > surv_max  # 所有被淘汰者都比所有存活者差
        else:
            combined = self._compute_combined_percent(scores, vote_shares)
            # 被淘汰者的综合百分比应该是最小的
            elim_min = np.min(combined[is_eliminated])
            surv_min = np.min(combined[~is_eliminated]) if np.any(~is_eliminated) else float('inf')
            return elim_min < surv_min
    
    def _validate_predictions(self, weekly_data: pd.DataFrame, 
                             elimination_info: pd.DataFrame):
        """验证预测准确率"""
        correct = 0
        bottom_n = 0
        total = 0
        success_count = 0
        
        for (season, week), week_result in self.week_results.items():
            # 不再跳过优化失败的周次，使用全部数据
            if week_result['success']:
                success_count += 1
            
            contestants = week_result['contestants']
            vote_shares = week_result['vote_shares']
            method = week_result['method']
            
            scores = np.array([c['score'] for c in contestants])
            is_eliminated = np.array([c['is_eliminated'] for c in contestants])
            n = len(contestants)
            n_eliminated = np.sum(is_eliminated)
            
            # 计算综合得分
            if method == 'rank':
                combined = self._compute_combined_rank(scores, vote_shares, n)
                # 排名法：预测综合排名数值最大的n_eliminated个人（排名最差）
                # np.argsort返回从小到大的索引，取最后 n_eliminated 个
                pred_indices = np.argsort(combined)[-n_eliminated:]
            else:
                combined = self._compute_combined_percent(scores, vote_shares)
                # 预测综合百分比最低的为淘汰者
                pred_indices = np.argsort(combined)[:n_eliminated]
            
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            if pred_eliminated == actual_eliminated:
                correct += 1
            
            # 检查是否在底部
            if method == 'rank':
                bottom_indices = np.argsort(combined)[-max(2, n_eliminated):]
            else:
                bottom_indices = np.argsort(combined)[:max(2, n_eliminated)]
            
            bottom_names = set(contestants[i]['name'] for i in bottom_indices)
            if actual_eliminated.issubset(bottom_names):
                bottom_n += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = bottom_n / total if total > 0 else 0
        print(f"\n============================================================")
        print(f"投票反推拟合验证（全部数据）")
        print(f"============================================================")
        print(f"检验周次数: {total}  （其中优化成功: {success_count}, 备用估计: {total - success_count}）")
        print(f"正确反推数: {correct}")
        print(f"反推准确率: {accuracy:.2%}")
        print(f"底N准确率: {bottom_accuracy:.2%}")
        print(f"")
        print(f"注: 此准确率衡量模型对已知淘汰结果的拟合能力")
        print(f"    (仅统计约束优化成功的周次)")
        print(f"============================================================")

        # 返回结果供外部使用
        return {
            'accuracy': accuracy,
            'bottom_accuracy': bottom_accuracy,
            'total': total,
            'correct': correct,
            'bottom_correct': bottom_n
        }
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰结果（用于验证）"""
        return self._validate_predictions(weekly_data, elimination_info)
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计结果"""
        return self.results_df
    
    def get_estimates_dict(self) -> Dict:
        """返回字典格式的估计结果（使用vote_share以保持精度）"""
        estimates = {}
        
        for (season, week), group in self.results_df.groupby(['season', 'week']):
            names = group['celebrity'].tolist()
            scores = group['total_score'].tolist()
            # 使用vote_share而不是estimated_votes，避免四舍五入导致的排名变化
            votes = group['vote_share'].tolist()
            popularities = group['popularity'].tolist() if 'popularity' in group.columns else [0] * len(names)
            
            estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes,
                'popularities': popularities
            }
        
        return estimates
    
    def get_samples_dict(self) -> Dict:
        """返回样本字典"""
        samples_dict = {}
        
        for _, row in self.results_df.iterrows():
            if pd.isna(row['estimated_votes']):
                continue
            key = (row['season'], row['week'], row['celebrity'])
            votes = row['estimated_votes']
            cv = 0.1 + 0.1 * (1 - row.get('vote_share', 0.5))
            samples = np.random.normal(votes, votes * cv, 100)
            samples = np.maximum(samples, 0)
            samples_dict[key] = samples
        
        return samples_dict
