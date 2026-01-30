"""
淘汰概率模型
使用淘汰结果作为观测数据，推断观众投票

核心思路：
1. 建立 log_votes = β₀ + β_score × score + effects 的投票模型
2. 根据赛季使用不同的投票合并方法：
   - 排名法（赛季1-2, 28-34）：评委排名 + 观众排名 → 综合排名
   - 百分比法（赛季3-27）：评委百分比 + 观众百分比 → 综合百分比
3. 淘汰概率与综合得分负相关
4. 使用实际淘汰结果来学习参数
5. 推断每个选手的投票数量
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from scipy.optimize import minimize
from scipy.special import softmax
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

# 定义两种投票方法适用的赛季
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))  # 排名法赛季: 1-2, 28-34
SEASONS_PERCENT_METHOD = list(range(3, 28))  # 百分比法赛季: 3-27


class EliminationProbabilityModel:
    """
    基于淘汰概率的投票估计模型
    
    模型结构：
    1. 投票模型：log(V_i) = β₀ + β_score × S_i + β_age × Age_i + α_partner + γ_season + δ_industry + ε
    2. 淘汰概率：P(淘汰_i | week) = softmax(-log(V_i))  (投票最低者最可能被淘汰)
    3. 似然函数：L = Π P(实际被淘汰者被淘汰)
    
    使用全部数据训练，最大化淘汰结果的似然
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数
        self.beta_0 = 10.0        # log投票基准 (e^10 ≈ 22000票)
        self.beta_score = 0.5     # 评分效应（正数：高分→高票）
        self.beta_age = 0.0       # 年龄效应
        self.sigma = 0.5          # 噪声标准差
        
        # 随机效应
        self.partner_effects = {}
        self.season_effects = {}
        self.industry_effects = {}
        
        # 编码映射
        self.partner_to_idx = {}
        self.season_to_idx = {}
        self.industry_to_idx = {}
        
        # 标准化参数
        self.score_mean = 0
        self.score_std = 1
        self.age_mean = 0
        self.age_std = 1
        
        # 结果存储
        self.vote_estimates = {}
        self.elimination_probs = {}
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """
        使用所有数据训练模型
        
        Args:
            weekly_data: 周级别数据
            elimination_info: 淘汰信息
        """
        print("正在训练淘汰概率模型...")
        
        # 准备训练数据
        train_data = self._prepare_training_data(weekly_data, elimination_info)
        
        # 编码分类变量
        self._encode_categorical(weekly_data)
        
        # 优化参数
        self._optimize_parameters(train_data, weekly_data)
        
        # 估计所有选手的投票
        self._estimate_all_votes(weekly_data)
        
        self.is_fitted = True
        print("模型训练完成!")
        
    def _prepare_training_data(self, weekly_data: pd.DataFrame, 
                               elimination_info: pd.DataFrame) -> List[Dict]:
        """
        准备训练数据：每周的选手列表和淘汰者
        """
        # 标准化评分和年龄
        self.score_mean = weekly_data['total_score'].mean()
        self.score_std = weekly_data['total_score'].std()
        self.age_mean = weekly_data['celebrity_age'].mean()
        self.age_std = weekly_data['celebrity_age'].std()
        
        train_data = []
        
        # 按赛季-周次分组
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            # 找到该周的淘汰者
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = elim['eliminated_name'].tolist()
            contestants = group['celebrity_name'].tolist()
            
            # 只处理淘汰者在选手列表中的情况
            valid_eliminated = [name for name in eliminated_names if name in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            # 获取选手数据（包含原始评分用于综合得分计算）
            contestants_data = []
            for _, row in group.iterrows():
                age_val = row['celebrity_age']
                contestants_data.append({
                    'name': row['celebrity_name'],
                    'score': (row['total_score'] - self.score_mean) / self.score_std,  # 标准化评分（用于模型）
                    'raw_score': row['total_score'],  # 原始评分（用于综合得分计算）
                    'age': (age_val - self.age_mean) / self.age_std if pd.notna(age_val) else 0,
                    'partner': row.get('ballroom_partner', 'Unknown'),
                    'industry': row.get('celebrity_industry', 'Unknown'),
                    'season': season,
                    'is_eliminated': row['celebrity_name'] in valid_eliminated
                })
            
            train_data.append({
                'season': season,
                'week': week,
                'contestants': contestants_data,
                'eliminated_names': valid_eliminated
            })
        
        print(f"  准备了 {len(train_data)} 个周次的训练数据")
        print(f"  - 排名法赛季周次: {sum(1 for d in train_data if d['season'] in SEASONS_RANK_METHOD)}")
        print(f"  - 百分比法赛季周次: {sum(1 for d in train_data if d['season'] in SEASONS_PERCENT_METHOD)}")
        return train_data
    
    def _encode_categorical(self, weekly_data: pd.DataFrame):
        """编码分类变量"""
        # 舞伴
        partners = weekly_data['ballroom_partner'].dropna().unique()
        self.partner_to_idx = {p: i for i, p in enumerate(partners)}
        
        # 赛季
        seasons = weekly_data['season'].unique()
        self.season_to_idx = {s: i for i, s in enumerate(seasons)}
        
        # 行业
        if 'celebrity_industry' in weekly_data.columns:
            industries = weekly_data['celebrity_industry'].dropna().unique()
            self.industry_to_idx = {ind: i for i, ind in enumerate(industries)}
        else:
            self.industry_to_idx = {'Unknown': 0}
    
    def _compute_log_votes(self, contestants_data: List[Dict], params: np.ndarray) -> np.ndarray:
        """
        计算选手的log投票
        
        params: [beta_0, beta_score, beta_age, sigma, 
                 partner_effects..., season_effects..., industry_effects...]
        """
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        beta_0 = params[0]
        beta_score = params[1]
        beta_age = params[2]
        # sigma = params[3]  # 用于不确定性，优化时不直接使用
        
        partner_effects = params[4:4+n_partners]
        season_effects = params[4+n_partners:4+n_partners+n_seasons]
        industry_effects = params[4+n_partners+n_seasons:4+n_partners+n_seasons+n_industries]
        
        log_votes = []
        for c in contestants_data:
            lv = beta_0 + beta_score * c['score'] + beta_age * c['age']
            
            # 添加随机效应
            if c['partner'] in self.partner_to_idx:
                lv += partner_effects[self.partner_to_idx[c['partner']]]
            if c['season'] in self.season_to_idx:
                lv += season_effects[self.season_to_idx[c['season']]]
            if c.get('industry', 'Unknown') in self.industry_to_idx:
                lv += industry_effects[self.industry_to_idx[c.get('industry', 'Unknown')]]
            
            log_votes.append(lv)
        
        return np.array(log_votes)
    
    def _compute_combined_score(self, judge_scores: np.ndarray, votes: np.ndarray, 
                                 season: int) -> np.ndarray:
        """
        根据赛季计算综合得分
        
        排名法（赛季1-2, 28-34）：
            combined_rank = judge_rank + fan_rank
            得分越低越好
            
        百分比法（赛季3-27）：
            combined_pct = judge_pct + fan_pct
            得分越高越好
        
        返回：综合得分（越高越好，即淘汰概率越低）
        """
        n = len(judge_scores)
        
        if season in SEASONS_RANK_METHOD:
            # 排名法：排名越小越好
            # 计算排名（得分高→排名小）
            judge_ranks = stats.rankdata(-judge_scores)  # 高分→低排名
            vote_ranks = stats.rankdata(-votes)          # 高票→低排名
            combined_ranks = judge_ranks + vote_ranks
            # 转换为得分（排名低→得分高）
            combined_score = -combined_ranks
        else:
            # 百分比法：百分比越高越好
            judge_total = np.sum(judge_scores)
            vote_total = np.sum(votes)
            
            if judge_total > 0:
                judge_pct = judge_scores / judge_total
            else:
                judge_pct = np.ones(n) / n
                
            if vote_total > 0:
                vote_pct = votes / vote_total
            else:
                vote_pct = np.ones(n) / n
            
            combined_score = judge_pct + vote_pct
        
        return combined_score
    
    def _compute_elimination_prob(self, log_votes: np.ndarray, judge_scores: np.ndarray,
                                   season: int, temperature: float = 1.0) -> np.ndarray:
        """
        计算淘汰概率
        
        使用综合得分：得分越低，淘汰概率越高
        """
        votes = np.exp(log_votes)
        combined_score = self._compute_combined_score(judge_scores, votes, season)
        
        # 使用softmax(-combined_score)：得分越低，淘汰概率越高
        neg_scores = -combined_score / temperature
        neg_scores = neg_scores - np.max(neg_scores)  # 数值稳定性
        probs = softmax(neg_scores)
        return probs
    
    def _negative_log_likelihood(self, params: np.ndarray, train_data: List[Dict]) -> float:
        """
        计算负对数似然
        
        目标：最大化实际被淘汰者的淘汰概率
        使用综合得分（评委+观众）来计算淘汰概率
        """
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        nll = 0.0
        
        # 正则化项（防止过拟合）
        beta_score = params[1]
        beta_age = params[2]
        partner_effects = params[4:4+n_partners]
        season_effects = params[4+n_partners:4+n_partners+n_seasons]
        industry_effects = params[4+n_partners+n_seasons:]
        
        # L2正则化
        reg_strength = 0.1
        nll += reg_strength * (beta_score**2 + beta_age**2)
        nll += reg_strength * np.sum(partner_effects**2)
        nll += reg_strength * np.sum(season_effects**2)
        nll += reg_strength * np.sum(industry_effects**2)
        
        for week_data in train_data:
            contestants = week_data['contestants']
            eliminated_names = week_data['eliminated_names']
            season = week_data['season']
            
            if len(contestants) < 2:
                continue
            
            # 计算log投票
            log_votes = self._compute_log_votes(contestants, params)
            
            # 获取原始评委评分（用于综合得分计算）
            judge_scores = np.array([c['raw_score'] for c in contestants])
            
            # 计算淘汰概率（使用综合得分）
            elim_probs = self._compute_elimination_prob(log_votes, judge_scores, season)
            
            # 累加被淘汰者的负对数概率
            for i, c in enumerate(contestants):
                if c['name'] in eliminated_names:
                    # 避免log(0)
                    prob = max(elim_probs[i], 1e-10)
                    nll -= np.log(prob)
        
        return nll
    
    def _optimize_parameters(self, train_data: List[Dict], weekly_data: pd.DataFrame):
        """
        优化模型参数
        """
        print("  正在优化参数（最大化淘汰似然）...")
        
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        # 初始参数
        # [beta_0, beta_score, beta_age, sigma, partner_effects, season_effects, industry_effects]
        n_params = 4 + n_partners + n_seasons + n_industries
        x0 = np.zeros(n_params)
        x0[0] = 10.0   # beta_0
        x0[1] = 0.5    # beta_score (正数：高分→高票→低淘汰概率)
        x0[2] = 0.0    # beta_age
        x0[3] = 0.5    # sigma
        
        # 参数边界
        bounds = [(5, 15)]           # beta_0
        bounds += [(0.01, 5.0)]      # beta_score (强制为正)
        bounds += [(-1, 1)]          # beta_age
        bounds += [(0.1, 2.0)]       # sigma
        bounds += [(-2, 2)] * n_partners    # partner effects
        bounds += [(-1, 1)] * n_seasons     # season effects
        bounds += [(-2, 2)] * n_industries  # industry effects
        
        # 优化
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(train_data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        
        if result.success:
            print(f"  优化成功! 最终损失: {result.fun:.4f}")
        else:
            print(f"  优化警告: {result.message}")
        
        # 提取参数
        params = result.x
        self.beta_0 = params[0]
        self.beta_score = params[1]
        self.beta_age = params[2]
        self.sigma = params[3]
        
        # 提取随机效应
        idx = 4
        for partner, i in self.partner_to_idx.items():
            self.partner_effects[partner] = params[idx + i]
        idx += n_partners
        
        for season, i in self.season_to_idx.items():
            self.season_effects[season] = params[idx + i]
        idx += n_seasons
        
        for industry, i in self.industry_to_idx.items():
            self.industry_effects[industry] = params[idx + i]
        
        # 打印参数
        print(f"\n  模型参数:")
        print(f"    β₀ = {self.beta_0:.3f} (基准log投票 ≈ {np.exp(self.beta_0):.0f}票)")
        print(f"    β_score = {self.beta_score:.3f} (评分效应，正数表示高分→高票)")
        print(f"    β_age = {self.beta_age:.3f} (年龄效应)")
        print(f"    σ = {self.sigma:.3f}")
        
        # 显示最有影响力的舞伴
        sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  舞伴效应 (Top 5):")
        for p, e in sorted_partners[:5]:
            print(f"    {p}: {e:+.3f}")
        
        # 显示最有影响力的行业
        sorted_industries = sorted(self.industry_effects.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  行业效应 (Top 5):")
        for ind, e in sorted_industries[:5]:
            print(f"    {ind}: {e:+.3f}")
    
    def _estimate_all_votes(self, weekly_data: pd.DataFrame):
        """
        估计所有选手的投票数量
        """
        print("\n  正在估计所有选手的投票...")
        
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            week_votes = []
            judge_scores_list = []
            
            for _, row in group.iterrows():
                # 标准化特征
                score_norm = (row['total_score'] - self.score_mean) / self.score_std
                age_val = row['celebrity_age']
                age_norm = (age_val - self.age_mean) / self.age_std if pd.notna(age_val) else 0
                
                # 计算log投票
                log_vote = self.beta_0 + self.beta_score * score_norm + self.beta_age * age_norm
                
                # 添加随机效应
                partner = row.get('ballroom_partner', 'Unknown')
                if partner in self.partner_effects:
                    log_vote += self.partner_effects[partner]
                
                if season in self.season_effects:
                    log_vote += self.season_effects[season]
                
                industry = row.get('celebrity_industry', 'Unknown')
                if industry in self.industry_effects:
                    log_vote += self.industry_effects[industry]
                
                # 转换为实际投票数
                votes = np.exp(log_vote)
                
                # 生成样本用于不确定性估计
                vote_samples = np.exp(np.random.normal(log_vote, self.sigma, 100))
                
                week_votes.append({
                    'season': season,
                    'week': week,
                    'celebrity': row['celebrity_name'],
                    'total_score': row['total_score'],
                    'estimated_votes': votes,
                    'log_votes': log_vote,
                    'vote_std': np.std(vote_samples),
                    'vote_ci_low': np.percentile(vote_samples, 2.5),
                    'vote_ci_high': np.percentile(vote_samples, 97.5),
                    'method': 'rank' if season in SEASONS_RANK_METHOD else 'percent'
                })
                judge_scores_list.append(row['total_score'])
                
                # 存储样本
                key = (season, week, row['celebrity_name'])
                self.vote_estimates[key] = votes
            
            # 计算该周的淘汰概率（使用综合得分）
            log_votes = np.array([v['log_votes'] for v in week_votes])
            judge_scores = np.array(judge_scores_list)
            elim_probs = self._compute_elimination_prob(log_votes, judge_scores, season)
            
            # 计算综合得分
            votes_array = np.exp(log_votes)
            combined_scores = self._compute_combined_score(judge_scores, votes_array, season)
            
            for i, v in enumerate(week_votes):
                v['elimination_prob'] = elim_probs[i]
                v['combined_score'] = combined_scores[i]
                results.append(v)
        
        self.results_df = pd.DataFrame(results)
        print(f"  完成 {len(results)} 条投票估计")
        print(f"  - 排名法: {len(self.results_df[self.results_df['method']=='rank'])} 条")
        print(f"  - 百分比法: {len(self.results_df[self.results_df['method']=='percent'])} 条")
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """
        预测淘汰结果并计算准确率
        使用综合得分（评委+观众）来预测
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()")
        
        correct = 0
        correct_rank = 0
        correct_percent = 0
        in_bottom_2 = 0
        total = 0
        total_rank = 0
        total_percent = 0
        
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            # 获取实际淘汰者
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            actual_eliminated = elim['eliminated_name'].tolist()
            contestants = group['celebrity_name'].tolist()
            
            # 检查淘汰者是否在选手列表中
            valid_eliminated = [n for n in actual_eliminated if n in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            # 获取该周的投票估计（按综合得分排序，低分在前）
            week_results = self.results_df[
                (self.results_df['season'] == season) & 
                (self.results_df['week'] == week)
            ].sort_values('combined_score', ascending=True)  # 综合得分低的先被淘汰
            
            if len(week_results) == 0:
                continue
            
            # 预测淘汰者（综合得分最低的N个）
            n_eliminated = len(valid_eliminated)
            predicted_eliminated = week_results.head(n_eliminated)['celebrity'].tolist()
            
            # 检查准确率
            is_correct = set(predicted_eliminated) == set(valid_eliminated)
            
            # 检查是否在底部
            bottom_n = week_results.head(max(2, n_eliminated))['celebrity'].tolist()
            all_in_bottom = all(n in bottom_n for n in valid_eliminated)
            
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            
            if is_correct:
                correct += 1
                if method == 'rank':
                    correct_rank += 1
                else:
                    correct_percent += 1
            if all_in_bottom:
                in_bottom_2 += 1
            total += 1
            if method == 'rank':
                total_rank += 1
            else:
                total_percent += 1
            
            results.append({
                'season': season,
                'week': week,
                'method': method,
                'actual': valid_eliminated,
                'predicted': predicted_eliminated,
                'is_correct': is_correct,
                'in_bottom': all_in_bottom
            })
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = in_bottom_2 / total if total > 0 else 0
        rank_accuracy = correct_rank / total_rank if total_rank > 0 else 0
        percent_accuracy = correct_percent / total_percent if total_percent > 0 else 0
        
        print(f"\n============================================================")
        print(f"淘汰预测结果")
        print(f"============================================================")
        print(f"总周次数: {total}")
        print(f"正确预测数: {correct}")
        print(f"淘汰预测准确率: {accuracy:.2%}")
        print(f"底N预测准确率: {bottom_accuracy:.2%}")
        print(f"------------------------------------------------------------")
        print(f"按方法分组:")
        print(f"  排名法 (S1-2, S28-34): {correct_rank}/{total_rank} = {rank_accuracy:.2%}")
        print(f"  百分比法 (S3-27): {correct_percent}/{total_percent} = {percent_accuracy:.2%}")
        print(f"============================================================")
        
        return {
            'accuracy': accuracy,
            'bottom_accuracy': bottom_accuracy,
            'rank_accuracy': rank_accuracy,
            'percent_accuracy': percent_accuracy,
            'total': total,
            'correct': correct,
            'results': results
        }
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计结果"""
        return self.results_df
    
    def get_samples_dict(self) -> Dict:
        """返回用于不确定性分析的样本字典"""
        samples_dict = {}
        
        for _, row in self.results_df.iterrows():
            key = (row['season'], row['week'], row['celebrity'])
            # 生成样本
            log_vote = row['log_votes']
            samples = np.exp(np.random.normal(log_vote, self.sigma, 100))
            samples_dict[key] = samples
        
        return samples_dict
