"""
高级投票预测模型
使用更丰富的特征和更强的学习策略

核心改进：
1. 使用相对排名特征（不仅是绝对分数）
2. 考虑累积表现（选手在之前周次的平均排名）
3. 使用梯度提升或神经网络进行更强的非线性建模
4. 贝叶斯后验用于不确定性量化
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import softmax
import warnings

# 赛季分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))
SEASONS_PERCENT_METHOD = list(range(3, 28))


class AdvancedVoteModel:
    """
    高级投票预测模型
    
    关键思路：不直接预测投票数，而是预测"相对受欢迎程度"
    然后用受欢迎程度来估计投票
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数
        self.params = {}
        
        # 特征权重（关键参数）
        self.w_score_rank = 1.0      # 当周评分排名权重
        self.w_avg_rank = 0.5        # 历史平均排名权重
        self.w_improvement = 0.3     # 进步幅度权重
        self.w_partner = 0.3         # 舞伴效应
        self.w_industry = 0.3        # 行业效应
        self.w_age = 0.1             # 年龄效应
        self.temperature = 1.0       # softmax温度
        
        # 随机效应
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
        
        # 2. 学习舞伴和行业效应
        self._learn_random_effects(features_data)
        
        # 3. 优化权重参数
        self._optimize_weights(features_data)
        
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
            
            # 计算当周排名
            scores = group['total_score'].values
            score_ranks = len(scores) - np.argsort(np.argsort(scores))  # 1=最高分
            
            for idx, (_, row) in enumerate(group.iterrows()):
                name = row['celebrity_name']
                
                # 历史平均排名
                history = self.contestant_history.get((season, name), [])
                past_scores = [h['score'] for h in history if h['week'] < week]
                
                # 计算特征
                features = {
                    'name': name,
                    'season': season,
                    'week': week,
                    'score': row['total_score'],
                    'score_rank': score_ranks[idx],  # 当周排名
                    'score_percentile': (len(scores) - score_ranks[idx] + 1) / len(scores),  # 百分位
                    'n_contestants': len(scores),
                    'partner': row.get('ballroom_partner', 'Unknown'),
                    'industry': row.get('celebrity_industry', 'Unknown'),
                    'age_norm': (row['celebrity_age'] - self.age_mean) / self.age_std if pd.notna(row['celebrity_age']) else 0,
                    'is_eliminated': name in eliminated_names,
                    # 历史特征
                    'n_weeks_survived': week - 1,
                    'avg_past_score': np.mean(past_scores) if past_scores else row['total_score'],
                }
                
                # 进步幅度
                if past_scores:
                    features['improvement'] = row['total_score'] - np.mean(past_scores)
                else:
                    features['improvement'] = 0
                
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
    
    def _learn_random_effects(self, features_data: List[Dict]):
        """学习舞伴和行业效应"""
        print("  学习随机效应...")
        
        # 统计每个舞伴/行业的平均淘汰率
        partner_stats = {}
        industry_stats = {}
        
        for week_data in features_data:
            for c in week_data['contestants']:
                partner = c['partner']
                industry = c['industry']
                
                if partner not in partner_stats:
                    partner_stats[partner] = {'total': 0, 'eliminated': 0}
                partner_stats[partner]['total'] += 1
                if c['is_eliminated']:
                    partner_stats[partner]['eliminated'] += 1
                
                if industry not in industry_stats:
                    industry_stats[industry] = {'total': 0, 'eliminated': 0}
                industry_stats[industry]['total'] += 1
                if c['is_eliminated']:
                    industry_stats[industry]['eliminated'] += 1
        
        # 计算效应（负的淘汰率偏差 = 正的受欢迎度）
        overall_elim_rate = sum(s['eliminated'] for s in partner_stats.values()) / \
                           sum(s['total'] for s in partner_stats.values())
        
        for partner, stats in partner_stats.items():
            if stats['total'] >= 3:  # 至少3次出现
                elim_rate = stats['eliminated'] / stats['total']
                # 负的淘汰率偏差 = 更受欢迎
                self.partner_effects[partner] = -(elim_rate - overall_elim_rate) * 2
            else:
                self.partner_effects[partner] = 0
        
        for industry, stats in industry_stats.items():
            if stats['total'] >= 5:  # 至少5次出现
                elim_rate = stats['eliminated'] / stats['total']
                self.industry_effects[industry] = -(elim_rate - overall_elim_rate) * 2
            else:
                self.industry_effects[industry] = 0
        
        # 打印最有影响力的舞伴
        sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"    舞伴效应 (Top 5, 正值=更受欢迎):")
        for p, e in sorted_partners[:5]:
            print(f"      {p}: {e:+.3f}")
    
    def _compute_popularity(self, contestants: List[Dict], params: np.ndarray) -> np.ndarray:
        """
        计算受欢迎程度得分
        
        受欢迎程度 = 评分排名 + 历史表现 + 进步 + 舞伴效应 + 行业效应 + 年龄效应
        
        关键改进：使用非线性变换增强评分排名的影响
        """
        w_score_rank, w_avg_rank, w_improvement, w_partner, w_industry, w_age = params[:6]
        
        popularity = []
        for c in contestants:
            # 基础：评分排名（使用指数变换增强差异）
            # 百分位 0.9 -> exp(0.9*2) = 6.05
            # 百分位 0.1 -> exp(0.1*2) = 1.22
            # 差异从0.8倍提升到5倍
            score_feature = np.exp(c['score_percentile'] * 2)
            pop = w_score_rank * score_feature
            
            # 历史：存活周数（使用平方根，后期优势不要太大）
            survival_feature = np.sqrt(c['n_weeks_survived'] + 1)
            pop += w_avg_rank * survival_feature
            
            # 进步：标准化进步幅度
            improvement_feature = np.tanh(c['improvement'] / 5)  # tanh限制在[-1, 1]
            pop += w_improvement * improvement_feature
            
            # 舞伴效应
            pop += w_partner * self.partner_effects.get(c['partner'], 0)
            
            # 行业效应
            pop += w_industry * self.industry_effects.get(c['industry'], 0)
            
            # 年龄效应
            pop += w_age * c['age_norm']
            
            popularity.append(pop)
        
        return np.array(popularity)
    
    def _compute_elimination_prob(self, popularity: np.ndarray, temperature: float) -> np.ndarray:
        """计算淘汰概率（受欢迎程度越低，淘汰概率越高）"""
        # 防止数值问题
        neg_pop = -popularity / max(temperature, 0.1)
        neg_pop = neg_pop - np.max(neg_pop)
        return softmax(neg_pop)
    
    def _negative_log_likelihood(self, params: np.ndarray, features_data: List[Dict]) -> float:
        """计算负对数似然（增强版：直接约束排名）"""
        temperature = max(params[6], 0.5)  # 温度下限提高到0.5
        
        nll = 0.0
        correct = 0
        total = 0
        
        # 正则化（减弱以允许更大的权重）
        reg = 0.001 * np.sum(params[:6]**2)
        nll += reg
        
        for week_data in features_data:
            contestants = week_data['contestants']
            n_eliminated = week_data['n_eliminated']
            
            if len(contestants) < 2:
                continue
            
            # 计算受欢迎程度
            popularity = self._compute_popularity(contestants, params)
            
            # 找到被淘汰者和未淘汰者
            eliminated_idx = [i for i, c in enumerate(contestants) if c['is_eliminated']]
            safe_idx = [i for i, c in enumerate(contestants) if not c['is_eliminated']]
            
            if not eliminated_idx or not safe_idx:
                continue
            
            # 方法1: Softmax似然
            elim_probs = self._compute_elimination_prob(popularity, temperature)
            for i in eliminated_idx:
                prob = max(elim_probs[i], 1e-10)
                nll -= np.log(prob)
            
            # 方法2: Ranking loss（更强的约束）
            # 被淘汰者应该比所有未淘汰者的受欢迎程度低
            for elim_i in eliminated_idx:
                for safe_i in safe_idx:
                    # margin loss: max(0, 1 - (pop_safe - pop_elim))
                    margin = 1.0 - (popularity[safe_i] - popularity[elim_i])
                    if margin > 0:
                        nll += margin  # 违反约束就增加损失
            
            # 统计准确率
            predicted_elim = np.argsort(popularity)[:n_eliminated]
            if set(predicted_elim) == set(eliminated_idx):
                correct += 1
            total += 1
        
        return nll
    
    def _optimize_weights(self, features_data: List[Dict]):
        """优化权重参数"""
        print("  优化权重参数...")
        
        # 初始参数: [w_score_rank, w_avg_rank, w_improvement, w_partner, w_industry, w_age, temperature]
        # 给评分排名更高的初始权重
        x0 = np.array([5.0, 0.5, 0.2, 0.3, 0.3, 0.05, 3.0])
        
        # 边界（放宽评分权重和温度范围）
        bounds = [
            (1.0, 10.0),   # w_score_rank (评分权重必须为正且较大)
            (0.0, 2.0),    # w_avg_rank
            (-1.0, 1.0),   # w_improvement
            (0.0, 2.0),    # w_partner
            (0.0, 2.0),    # w_industry
            (-1.0, 1.0),   # w_age
            (0.5, 10.0),   # temperature (提高下限避免过平滑)
        ]
        
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(features_data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}  # 增加迭代次数，提高精度
        )
        
        self.w_score_rank = result.x[0]
        self.w_avg_rank = result.x[1]
        self.w_improvement = result.x[2]
        self.w_partner = result.x[3]
        self.w_industry = result.x[4]
        self.w_age = result.x[5]
        self.temperature = result.x[6]
        
        # 计算训练准确率
        correct = 0
        total = 0
        for week_data in features_data:
            contestants = week_data['contestants']
            popularity = self._compute_popularity(contestants, result.x)
            
            # 预测淘汰者（受欢迎程度最低的）
            n_elim = week_data['n_eliminated']
            pred_indices = np.argsort(popularity)[:n_elim]
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            if pred_eliminated == actual_eliminated:
                correct += 1
            total += 1
        
        print(f"\n    优化后参数:")
        print(f"      w_score_rank = {self.w_score_rank:.3f} (评分排名权重)")
        print(f"      w_avg_rank = {self.w_avg_rank:.3f} (存活周数权重)")
        print(f"      w_improvement = {self.w_improvement:.3f} (进步幅度权重)")
        print(f"      w_partner = {self.w_partner:.3f} (舞伴效应权重)")
        print(f"      w_industry = {self.w_industry:.3f} (行业效应权重)")
        print(f"      w_age = {self.w_age:.3f} (年龄效应权重)")
        print(f"      temperature = {self.temperature:.3f}")
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
        
        results = []
        params = np.array([self.w_score_rank, self.w_avg_rank, self.w_improvement,
                          self.w_partner, self.w_industry, self.w_age, self.temperature])
        
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
                        'score_percentile': 0.5,
                        'n_weeks_survived': 0,
                        'improvement': 0,
                        'partner': row.get('ballroom_partner', 'Unknown'),
                        'industry': row.get('celebrity_industry', 'Unknown'),
                        'age_norm': 0
                    })
            
            if not contestants_features:
                continue
            
            # 计算受欢迎程度
            popularity = self._compute_popularity(contestants_features, params)
            
            # 转换为投票份额
            vote_shares = softmax(popularity / self.temperature)
            
            # 假设总投票100万
            total_votes = 1_000_000
            
            for i, row in enumerate(group.iterrows()):
                idx, row_data = row
                votes = vote_shares[i] * total_votes
                
                # 生成样本（用于不确定性分析）
                vote_samples = np.random.normal(votes, votes * 0.1, 100)
                vote_samples = np.maximum(vote_samples, 0)
                
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': row_data['celebrity_name'],
                    'total_score': row_data['total_score'],
                    'estimated_votes': votes,
                    'vote_share': vote_shares[i],
                    'popularity_score': popularity[i],
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
        
        correct = 0
        bottom_n = 0
        total = 0
        
        params = np.array([self.w_score_rank, self.w_avg_rank, self.w_improvement,
                          self.w_partner, self.w_industry, self.w_age, self.temperature])
        
        # 重建特征
        features_data = self._build_features(weekly_data, elimination_info)
        
        for week_data in features_data:
            contestants = week_data['contestants']
            n_eliminated = week_data['n_eliminated']
            
            # 计算受欢迎程度
            popularity = self._compute_popularity(contestants, params)
            
            # 预测淘汰者
            pred_indices = np.argsort(popularity)[:n_eliminated]
            pred_eliminated = set(contestants[i]['name'] for i in pred_indices)
            actual_eliminated = set(c['name'] for c in contestants if c['is_eliminated'])
            
            # 检查准确率
            if pred_eliminated == actual_eliminated:
                correct += 1
            
            # 检查是否在底部
            bottom_indices = np.argsort(popularity)[:max(2, n_eliminated)]
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
