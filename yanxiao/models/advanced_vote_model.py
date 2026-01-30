"""
高级投票预测模型
使用更丰富的特征和更强的学习策略

核心改进：
1. 使用相对排名特征（不仅是绝对分数）
2. 考虑累积表现（选手在之前周次的平均排名）
3. 区分排名法和百分比法的不同淘汰逻辑
4. 使用条件排名似然进行优化
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax
from scipy.stats import rankdata
import warnings

# 赛季分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))
SEASONS_PERCENT_METHOD = list(range(3, 28))


class AdvancedVoteModel:
    """
    高级投票预测模型
    
    关键思路：
    1. 模型学习"观众投票调整因子" = 实际排名 - 评分排名
    2. 被淘汰者的综合排名必须最低
    3. 使用条件排名似然：P(淘汰者综合排名最低 | 评分)
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数（投票调整因子的权重，共10个）
        self.w_base = 0.0            # 基准调整
        self.w_score_gap = 0.5       # 与最高分的差距效应
        self.w_weeks_survived = 0.3  # 存活周数效应（越久越受欢迎）
        self.w_improvement = 0.2     # 进步效应
        self.w_partner = 0.3         # 舞伴效应
        self.w_industry = 0.3        # 行业效应
        self.w_age = 0.0             # 年龄效应
        self.w_underdog = 0.2        # 黑马效应（低分高存活）
        self.w_bottom = 0.5          # 垫底惩罚
        self.w_recent = 0.3          # 最近趋势
        self.w_stability = 0.2       # 稳定性效应
        self.temperature = 0.5       # softmax温度（越小越确定）
        
        # 随机效应（通过回归学习）
        self.partner_effects = {}
        self.industry_effects = {}
        
        # 选手历史数据
        self.contestant_history = {}
        
        # 标准化参数
        self.age_mean = 0
        self.age_std = 1
        
        # 结果
        self.results_df = None
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """训练模型"""
        print("正在训练高级投票模型...")
        
        # 1. 构建特征
        features_data = self._build_features(weekly_data, elimination_info)
        
        # 2. 使用回归学习随机效应
        self._learn_effects_by_regression(features_data)
        
        # 3. 使用全局优化搜索最优参数
        self._optimize_weights_global(features_data)
        
        # 4. 估计投票
        self._estimate_votes(weekly_data, features_data)
        
        self.is_fitted = True
        print("模型训练完成!")
        
    def _build_features(self, weekly_data: pd.DataFrame, 
                        elimination_info: pd.DataFrame) -> List[Dict]:
        """构建丰富的特征"""
        print("  构建特征...")
        
        # 标准化年龄
        ages = weekly_data['celebrity_age'].dropna()
        self.age_mean = ages.mean()
        self.age_std = ages.std() if ages.std() > 0 else 1
        
        # 按选手构建历史
        self.contestant_history = {}
        for _, row in weekly_data.iterrows():
            name = row['celebrity_name']
            season = row['season']
            week = row['week']
            score = row['total_score']
            
            key = (season, name)
            if key not in self.contestant_history:
                self.contestant_history[key] = []
            self.contestant_history[key].append({
                'week': week,
                'score': score
            })
        
        # 构建每周特征
        features_data = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            # 获取淘汰者
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = set(elim['eliminated_name'].tolist())
            contestants = []
            
            # 计算当周排名和分数
            scores = group['total_score'].values
            n = len(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            
            # 评分排名（1=最高分）
            score_ranks = n - rankdata(scores, method='ordinal') + 1
            
            for idx, (_, row) in enumerate(group.iterrows()):
                name = row['celebrity_name']
                score = row['total_score']
                
                # 历史数据
                history = self.contestant_history.get((season, name), [])
                past_scores = [h['score'] for h in history if h['week'] < week]
                past_ranks = [h['rank'] for h in history if h['week'] < week]
                
                # 计算丰富的特征
                features = {
                    'name': name,
                    'season': season,
                    'week': week,
                    'score': score,
                    'score_rank': int(score_ranks[idx]),
                    'n_contestants': n,
                    # 与最高分的差距（标准化）
                    'score_gap': (max_score - score) / score_range,
                    # 评分百分位（0-1，1=最高）
                    'score_percentile': 1 - (score_ranks[idx] - 1) / max(n - 1, 1),
                    # 是否垫底
                    'is_bottom': 1 if score_ranks[idx] == n else 0,
                    # 是否倒数第二
                    'is_second_bottom': 1 if score_ranks[idx] == n - 1 else 0,
                    # 是否前三名
                    'is_top3': 1 if score_ranks[idx] <= 3 else 0,
                    # 舞伴和行业
                    'partner': row.get('ballroom_partner', 'Unknown'),
                    'industry': row.get('celebrity_industry', 'Unknown'),
                    # 年龄（标准化）
                    'age_norm': (row['celebrity_age'] - self.age_mean) / self.age_std if pd.notna(row['celebrity_age']) else 0,
                    # 存活周数（标准化）
                    'weeks_survived': (week - 1) / 10,
                    # 历史平均分
                    'avg_past_score': np.mean(past_scores) if past_scores else score,
                    # 历史平均排名
                    'avg_past_rank': np.mean(past_ranks) if past_ranks else score_ranks[idx],
                    # 进步幅度
                    'improvement': (score - np.mean(past_scores)) / score_range if past_scores else 0,
                    # 排名进步
                    'rank_improvement': (np.mean(past_ranks) - score_ranks[idx]) / n if past_ranks else 0,
                    # 是否为"黑马"（低分但存活多周）
                    'underdog_score': (week - 1) * (1 - (score - min_score) / score_range) / 10,
                    # 连续垫底次数
                    'consecutive_bottom': sum(1 for h in history[-3:] if h.get('rank', 0) == h.get('n', 0)),
                    # 最近趋势（最近3周的平均排名）
                    'recent_trend': np.mean([h['rank'] / h['n'] for h in history[-3:]]) if len(history) >= 1 else score_ranks[idx] / n,
                    # 稳定性（排名标准差）
                    'stability': np.std(past_ranks) / n if len(past_ranks) >= 2 else 0.5,
                    # 淘汰标签
                    'is_eliminated': name in eliminated_names,
                }
                
                contestants.append(features)
            
            # 只有当淘汰者在选手中才加入
            valid_eliminated = [c for c in contestants if c['is_eliminated']]
            if valid_eliminated:
                features_data.append({
                    'season': season,
                    'week': week,
                    'contestants': contestants,
                    'n_eliminated': len(valid_eliminated),
                    'method': 'rank' if season in SEASONS_RANK_METHOD else 'percent'
                })
        
        print(f"    构建了 {len(features_data)} 个周次的特征")
        return features_data
    
    def _learn_effects_by_regression(self, features_data: List[Dict]):
        """通过回归学习舞伴和行业效应"""
        print("  学习随机效应（回归方法）...")
        
        # 收集数据：被淘汰者 vs 未被淘汰者的特征差异
        partner_vote_boost = {}  # 舞伴带来的投票提升
        industry_vote_boost = {}
        
        for week_data in features_data:
            contestants = week_data['contestants']
            n = len(contestants)
            
            # 计算被淘汰者和未淘汰者的平均评分排名
            eliminated = [c for c in contestants if c['is_eliminated']]
            survived = [c for c in contestants if not c['is_eliminated']]
            
            if not eliminated or not survived:
                continue
            
            avg_elim_rank = np.mean([c['score_rank'] for c in eliminated])
            avg_surv_rank = np.mean([c['score_rank'] for c in survived])
            
            # 如果被淘汰者评分不是最低，说明有投票"救"了低分者
            for c in contestants:
                partner = c['partner']
                industry = c['industry']
                
                # 投票调整 = 实际结果 - 评分预期
                # 正值 = 获得额外投票支持
                if c['is_eliminated']:
                    # 被淘汰：投票低于评分预期
                    vote_effect = -(n - c['score_rank']) / n  # 负值
                else:
                    # 存活：投票高于或等于评分预期
                    vote_effect = (c['score_rank'] - 1) / n  # 存活奖励
                
                # 累积效应
                if partner not in partner_vote_boost:
                    partner_vote_boost[partner] = []
                partner_vote_boost[partner].append(vote_effect)
                
                if industry not in industry_vote_boost:
                    industry_vote_boost[industry] = []
                industry_vote_boost[industry].append(vote_effect)
        
        # 计算平均效应
        for partner, effects in partner_vote_boost.items():
            if len(effects) >= 3:
                self.partner_effects[partner] = np.mean(effects)
            else:
                self.partner_effects[partner] = 0
        
        for industry, effects in industry_vote_boost.items():
            if len(effects) >= 5:
                self.industry_effects[industry] = np.mean(effects)
            else:
                self.industry_effects[industry] = 0
        
        # 中心化效应
        if self.partner_effects:
            mean_partner = np.mean(list(self.partner_effects.values()))
            self.partner_effects = {k: v - mean_partner for k, v in self.partner_effects.items()}
        
        if self.industry_effects:
            mean_industry = np.mean(list(self.industry_effects.values()))
            self.industry_effects = {k: v - mean_industry for k, v in self.industry_effects.items()}
        
        # 打印Top效应
        sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    舞伴效应 (Top 5, 正值=更多投票支持):")
        for p, e in sorted_partners[:5]:
            print(f"      {p}: {e:+.3f}")
        
        sorted_industries = sorted(self.industry_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    行业效应 (Top 5):")
        for ind, e in sorted_industries[:5]:
            print(f"      {ind}: {e:+.3f}")
    
    def _compute_vote_adjustment(self, contestants: List[Dict], params: np.ndarray) -> np.ndarray:
        """
        计算投票调整因子
        
        正的调整 = 观众投票比评分预期更多（更受欢迎）
        负的调整 = 观众投票比评分预期更少
        
        最终综合得分 = 评分 + 投票调整
        综合得分最低者被淘汰
        """
        (w_score_gap, w_weeks, w_improve, w_partner, w_industry, w_age, 
         w_underdog, w_bottom, w_recent, w_stability) = params[:10]
        
        adjustments = []
        for c in contestants:
            adj = 0.0
            
            # 1. 与最高分的差距效应（评分越低，可能需要更多投票才能存活）
            adj -= w_score_gap * c['score_gap']
            
            # 2. 存活周数效应（存活越久，粉丝基础越大）
            adj += w_weeks * c['weeks_survived']
            
            # 3. 进步效应（进步明显可能获得更多支持）
            adj += w_improve * c['improvement']
            
            # 4. 舞伴效应
            adj += w_partner * self.partner_effects.get(c['partner'], 0)
            
            # 5. 行业效应
            adj += w_industry * self.industry_effects.get(c['industry'], 0)
            
            # 6. 年龄效应
            adj += w_age * c['age_norm']
            
            # 7. 黑马效应
            adj += w_underdog * c['underdog_score']
            
            # 8. 垫底惩罚（垫底更容易被淘汰）
            adj -= w_bottom * (c['is_bottom'] + 0.5 * c['is_second_bottom'])
            
            # 9. 最近趋势（最近表现不好→更可能被淘汰）
            adj -= w_recent * c['recent_trend']
            
            # 10. 稳定性效应（不稳定→观众不确定→可能少投票）
            adj -= w_stability * c['stability']
            
            adjustments.append(adj)
        
        return np.array(adjustments)
    
    def _compute_combined_score(self, contestants: List[Dict], params: np.ndarray, 
                                method: str) -> np.ndarray:
        """
        计算综合得分
        
        排名法：评分排名 + 投票排名（基于调整因子）
        百分比法：评分百分比 + 投票百分比（基于调整因子）
        """
        n = len(contestants)
        vote_adjustments = self._compute_vote_adjustment(contestants, params)
        
        # 投票排名（调整越高，排名越靠前=数字越小）
        vote_ranks = n - rankdata(vote_adjustments, method='ordinal') + 1
        
        if method == 'rank':
            # 排名法：评分排名 + 投票排名，越小越好
            score_ranks = np.array([c['score_rank'] for c in contestants])
            combined = score_ranks + vote_ranks
        else:
            # 百分比法：(评分百分比 + 投票百分比) / 2，越大越好
            score_pcts = np.array([c['score_percentile'] for c in contestants])
            vote_pcts = 1 - (vote_ranks - 1) / max(n - 1, 1)
            combined = -(score_pcts + vote_pcts)  # 负号使得越小=越可能淘汰
        
        return combined
    
    def _compute_accuracy(self, params: np.ndarray, features_data: List[Dict]) -> Tuple[int, int]:
        """计算准确率"""
        correct = 0
        total = 0
        
        for week_data in features_data:
            contestants = week_data['contestants']
            n_eliminated = week_data['n_eliminated']
            method = week_data['method']
            
            # 计算综合得分
            combined = self._compute_combined_score(contestants, params, method)
            
            # 预测淘汰者（综合得分最高的N个）
            pred_indices = np.argsort(combined)[-n_eliminated:]
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            if pred_eliminated == actual_eliminated:
                correct += 1
            total += 1
        
        return correct, total
    
    def _negative_accuracy(self, params: np.ndarray, features_data: List[Dict]) -> float:
        """负准确率（用于最大化）"""
        correct, total = self._compute_accuracy(params, features_data)
        return -correct / total if total > 0 else 0
    
    def _optimize_weights_global(self, features_data: List[Dict]):
        """使用全局优化搜索最优参数"""
        print("  全局优化搜索最优参数...")
        
        # 参数: [w_score_gap, w_weeks, w_improve, w_partner, w_industry, w_age, w_underdog, w_bottom, w_recent, w_stability]
        bounds = [
            (0.0, 5.0),    # w_score_gap
            (0.0, 3.0),    # w_weeks (存活周数)
            (-2.0, 2.0),   # w_improve (进步)
            (0.0, 5.0),    # w_partner
            (0.0, 5.0),    # w_industry
            (-2.0, 2.0),   # w_age
            (0.0, 3.0),    # w_underdog
            (0.0, 5.0),    # w_bottom (垫底惩罚)
            (0.0, 3.0),    # w_recent (最近趋势)
            (0.0, 2.0),    # w_stability (稳定性)
        ]
        
        # 使用差分进化全局优化
        result = differential_evolution(
            self._negative_accuracy,
            bounds,
            args=(features_data,),
            seed=self.random_seed,
            maxiter=300,
            tol=0.001,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        self.w_score_gap = result.x[0]
        self.w_weeks_survived = result.x[1]
        self.w_improvement = result.x[2]
        self.w_partner = result.x[3]
        self.w_industry = result.x[4]
        self.w_age = result.x[5]
        self.w_underdog = result.x[6]
        self.w_bottom = result.x[7]
        self.w_recent = result.x[8]
        self.w_stability = result.x[9]
        
        correct, total = self._compute_accuracy(result.x, features_data)
        
        print(f"\n    优化后参数:")
        print(f"      w_score_gap = {self.w_score_gap:.3f} (与最高分差距权重)")
        print(f"      w_weeks_survived = {self.w_weeks_survived:.3f} (存活周数权重)")
        print(f"      w_improvement = {self.w_improvement:.3f} (进步幅度权重)")
        print(f"      w_partner = {self.w_partner:.3f} (舞伴效应权重)")
        print(f"      w_industry = {self.w_industry:.3f} (行业效应权重)")
        print(f"      w_age = {self.w_age:.3f} (年龄效应权重)")
        print(f"      w_underdog = {self.w_underdog:.3f} (黑马效应权重)")
        print(f"      w_bottom = {self.w_bottom:.3f} (垫底惩罚权重)")
        print(f"      w_recent = {self.w_recent:.3f} (最近趋势权重)")
        print(f"      w_stability = {self.w_stability:.3f} (稳定性权重)")
        print(f"\n    训练准确率: {correct}/{total} = {correct/total:.2%}")
    
    def _estimate_votes(self, weekly_data: pd.DataFrame, features_data: List[Dict]):
        """估计投票数量"""
        print("\n  估计投票数量...")
        
        # 创建特征查找表
        feature_lookup = {}
        for week_data in features_data:
            for c in week_data['contestants']:
                key = (c['season'], c['week'], c['name'])
                feature_lookup[key] = c
        
        # 获取当前参数
        params = np.array([self.w_score_gap, self.w_weeks_survived, self.w_improvement,
                          self.w_partner, self.w_industry, self.w_age, self.w_underdog])
        
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            contestants_features = []
            for _, row in group.iterrows():
                key = (season, week, row['celebrity_name'])
                if key in feature_lookup:
                    contestants_features.append(feature_lookup[key])
                else:
                    # 创建基本特征
                    contestants_features.append({
                        'name': row['celebrity_name'],
                        'score': row['total_score'],
                        'score_rank': 1,
                        'score_gap': 0,
                        'score_percentile': 0.5,
                        'weeks_survived': 0,
                        'improvement': 0,
                        'underdog_score': 0,
                        'partner': row.get('ballroom_partner', 'Unknown'),
                        'industry': row.get('celebrity_industry', 'Unknown'),
                        'age_norm': 0
                    })
            
            if not contestants_features:
                continue
            
            n = len(contestants_features)
            
            # 计算投票调整因子
            vote_adjustments = self._compute_vote_adjustment(contestants_features, params)
            
            # 转换为投票份额
            # 基础份额来自评分
            score_shares = np.array([c['score_percentile'] for c in contestants_features])
            
            # 加上投票调整
            adjusted_shares = score_shares + vote_adjustments
            adjusted_shares = np.maximum(adjusted_shares, 0.01)  # 确保正数
            vote_shares = adjusted_shares / np.sum(adjusted_shares)
            
            # 假设总投票100万
            total_votes = 1_000_000
            
            for i, (_, row) in enumerate(group.iterrows()):
                votes = vote_shares[i] * total_votes
                
                # 生成样本（用于不确定性分析）
                cv = 0.1 + 0.1 * (1 - vote_shares[i])  # 份额越小，不确定性越大
                vote_samples = np.random.normal(votes, votes * cv, 100)
                vote_samples = np.maximum(vote_samples, 0)
                
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': row['celebrity_name'],
                    'total_score': row['total_score'],
                    'estimated_votes': votes,
                    'vote_share': vote_shares[i],
                    'vote_adjustment': vote_adjustments[i],
                    'vote_std': np.std(vote_samples),
                    'vote_ci_low': np.percentile(vote_samples, 2.5),
                    'vote_ci_high': np.percentile(vote_samples, 97.5)
                })
        
        self.results_df = pd.DataFrame(results)
        print(f"    完成 {len(results)} 条投票估计")
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰结果"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 重建特征
        features_data = self._build_features(weekly_data, elimination_info)
        
        # 获取当前参数（10个参数）
        params = np.array([
            self.w_score_gap, self.w_weeks_survived, self.w_improvement,
            self.w_partner, self.w_industry, self.w_age, self.w_underdog,
            self.w_bottom, self.w_recent, self.w_stability
        ])
        
        correct = 0
        bottom_n = 0
        total = 0
        
        for week_data in features_data:
            contestants = week_data['contestants']
            n_eliminated = week_data['n_eliminated']
            method = week_data['method']
            n = len(contestants)
            
            # 计算综合得分
            combined = self._compute_combined_score(contestants, params, method)
            
            # 预测淘汰者（综合得分最高的N个，因为combined越大越可能淘汰）
            pred_indices = np.argsort(combined)[-n_eliminated:]
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            # 检查准确率
            if pred_eliminated == actual_eliminated:
                correct += 1
            
            # 检查是否在底部
            bottom_indices = np.argsort(combined)[-max(2, n_eliminated):]
            bottom_names = set(contestants[i]['name'] for i in bottom_indices)
            if actual_eliminated.issubset(bottom_names):
                bottom_n += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = bottom_n / total if total > 0 else 0
        
        print(f"\n============================================================")
        print(f"淘汰预测结果")
        print(f"============================================================")
        print(f"总周次数: {total}")
        print(f"正确预测数: {correct}")
        print(f"淘汰预测准确率: {accuracy:.2%}")
        print(f"底N预测准确率: {bottom_accuracy:.2%}")
        print(f"============================================================")
        
        return {
            'accuracy': accuracy,
            'bottom_accuracy': bottom_accuracy,
            'total': total,
            'correct': correct
        }
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计结果"""
        return self.results_df
    
    def get_estimates_dict(self) -> Dict:
        """返回字典格式的估计结果（用于一致性检验）"""
        estimates = {}
        
        for (season, week), group in self.results_df.groupby(['season', 'week']):
            names = group['celebrity'].tolist()
            scores = group['total_score'].tolist()
            votes = group['estimated_votes'].tolist()
            
            estimates[(season, week)] = {
                'names': names,
                'scores': scores,
                'votes': votes
            }
        
        return estimates
    
    def get_samples_dict(self) -> Dict:
        """返回样本字典（用于不确定性分析）"""
        samples_dict = {}
        
        for _, row in self.results_df.iterrows():
            key = (row['season'], row['week'], row['celebrity'])
            # 生成样本
            votes = row['estimated_votes']
            samples = np.random.normal(votes, votes * 0.1, 100)
            samples = np.maximum(samples, 0)
            samples_dict[key] = samples
        
        return samples_dict
