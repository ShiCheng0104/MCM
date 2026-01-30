"""
观众投票估计模型（改进版）

核心思路：
1. 投票模型：log(V_i) = β₀ + β_score × S_i + 随机效应 + ε
2. 淘汰规则：综合得分最低者被淘汰
3. 似然函数：P(被淘汰者的综合得分最低)

关键改进：
- 使用正确的排名法/百分比法计算综合得分
- 似然函数基于"谁的综合得分最低"
- 参数学习目标是最大化淘汰结果的可解释性
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata
import warnings

# 投票合并方法的赛季划分
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))
SEASONS_PERCENT_METHOD = list(range(3, 28))


class VoteModel:
    """
    基于淘汰结果的投票估计模型
    
    核心假设：
    - 投票与评分正相关：高分选手倾向于获得更多投票
    - 舞伴效应：优秀舞伴带来额外投票
    - 行业效应：某些行业背景的选手更受欢迎
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数
        self.beta_0 = 10.0        # log投票基准
        self.beta_score = 1.0     # 评分效应（关键参数）
        self.beta_age = 0.0       # 年龄效应
        self.sigma = 0.5          # 噪声
        
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
        
        # 列名
        self.name_col = 'celebrity_name'
        self.age_col = 'celebrity_age'
        self.partner_col = 'ballroom_partner'
        self.industry_col = 'celebrity_industry'
        
        # 结果
        self.results_df = None
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """训练模型"""
        print("正在训练投票模型...")
        
        # 准备数据
        train_data = self._prepare_data(weekly_data, elimination_info)
        print(f"  有效训练周次: {len(train_data)}")
        
        # 编码分类变量
        self._encode_categorical(weekly_data)
        
        # 优化参数
        self._optimize_parameters(train_data)
        
        # 估计投票
        self._estimate_votes(weekly_data)
        
        self.is_fitted = True
        print("模型训练完成!")
        
    def _prepare_data(self, weekly_data: pd.DataFrame, 
                      elimination_info: pd.DataFrame) -> List[Dict]:
        """准备训练数据"""
        # 检测列名
        if 'celebrity_name' in weekly_data.columns:
            self.name_col = 'celebrity_name'
        else:
            self.name_col = 'celebrity'
            
        if 'celebrity_age' in weekly_data.columns:
            self.age_col = 'celebrity_age'
        elif 'age' in weekly_data.columns:
            self.age_col = 'age'
        else:
            self.age_col = None
            
        if 'ballroom_partner' in weekly_data.columns:
            self.partner_col = 'ballroom_partner'
        else:
            self.partner_col = 'partner'
            
        if 'celebrity_industry' in weekly_data.columns:
            self.industry_col = 'celebrity_industry'
        else:
            self.industry_col = 'industry'
        
        # 计算标准化参数
        self.score_mean = weekly_data['total_score'].mean()
        self.score_std = weekly_data['total_score'].std()
        
        if self.age_col and self.age_col in weekly_data.columns:
            self.age_mean = weekly_data[self.age_col].mean()
            self.age_std = weekly_data[self.age_col].std()
        
        train_data = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            # 找淘汰者
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
                
            eliminated_names = elim['eliminated_name'].tolist()
            contestants = group[self.name_col].tolist()
            
            valid_eliminated = [n for n in eliminated_names if n in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            # 选手数据
            contestants_data = []
            for _, row in group.iterrows():
                age_val = row[self.age_col] if self.age_col else 30
                contestants_data.append({
                    'name': row[self.name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'age_norm': (age_val - self.age_mean) / self.age_std if (self.age_col and pd.notna(age_val)) else 0,
                    'partner': row.get(self.partner_col, 'Unknown') if self.partner_col in row.index else 'Unknown',
                    'industry': row.get(self.industry_col, 'Unknown') if self.industry_col in row.index else 'Unknown',
                    'season': season,
                    'is_eliminated': row[self.name_col] in valid_eliminated
                })
            
            train_data.append({
                'season': season,
                'week': week,
                'contestants': contestants_data,
                'eliminated_names': valid_eliminated,
                'method': 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            })
        
        return train_data
    
    def _encode_categorical(self, weekly_data: pd.DataFrame):
        """编码分类变量"""
        if self.partner_col in weekly_data.columns:
            partners = weekly_data[self.partner_col].dropna().unique()
            self.partner_to_idx = {p: i for i, p in enumerate(partners)}
        
        seasons = weekly_data['season'].unique()
        self.season_to_idx = {s: i for i, s in enumerate(seasons)}
        
        if self.industry_col in weekly_data.columns:
            industries = weekly_data[self.industry_col].dropna().unique()
            self.industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    
    def _compute_log_votes(self, contestants: List[Dict], params: Dict) -> np.ndarray:
        """计算选手的log投票"""
        log_votes = []
        
        for c in contestants:
            lv = params['beta_0'] + params['beta_score'] * c['score_norm']
            lv += params['beta_age'] * c['age_norm']
            
            # 随机效应
            if c['partner'] in self.partner_to_idx:
                idx = self.partner_to_idx[c['partner']]
                if idx < len(params['partner_eff']):
                    lv += params['partner_eff'][idx]
            
            if c['season'] in self.season_to_idx:
                idx = self.season_to_idx[c['season']]
                if idx < len(params['season_eff']):
                    lv += params['season_eff'][idx]
            
            if c.get('industry', 'Unknown') in self.industry_to_idx:
                idx = self.industry_to_idx[c['industry']]
                if idx < len(params['industry_eff']):
                    lv += params['industry_eff'][idx]
            
            log_votes.append(lv)
        
        return np.array(log_votes)
    
    def _compute_combined_score(self, scores: np.ndarray, log_votes: np.ndarray, 
                                method: str) -> np.ndarray:
        """
        计算综合得分
        
        排名法：综合排名 = 评委排名 + 投票排名（越小越好）
        百分比法：综合百分比 = 0.5 * 评委百分比 + 0.5 * 投票百分比（越大越好）
        
        返回：淘汰得分（越高越可能被淘汰）
        """
        n = len(scores)
        
        if method == 'rank':
            # 排名法：排名越小越好
            score_ranks = rankdata(-scores)  # 高分=小排名
            vote_ranks = rankdata(-log_votes)  # 高票=小排名
            combined_ranks = score_ranks + vote_ranks
            # 综合排名越大，越可能被淘汰
            return combined_ranks
        else:
            # 百分比法
            score_pct = scores / scores.sum() if scores.sum() > 0 else np.ones(n) / n
            votes = np.exp(log_votes)
            vote_pct = votes / votes.sum() if votes.sum() > 0 else np.ones(n) / n
            combined_pct = 0.5 * score_pct + 0.5 * vote_pct
            # 百分比越小，越可能被淘汰（取负数）
            return -combined_pct
    
    def _compute_elimination_prob(self, elim_scores: np.ndarray, 
                                  temperature: float = 1.0) -> np.ndarray:
        """计算淘汰概率（elim_score越高越可能被淘汰）"""
        return softmax(elim_scores / temperature)
    
    def _negative_log_likelihood(self, params_array: np.ndarray, 
                                 train_data: List[Dict]) -> float:
        """负对数似然"""
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        # 解析参数
        params = {
            'beta_0': params_array[0],
            'beta_score': params_array[1],
            'beta_age': params_array[2],
            'sigma': max(params_array[3], 0.1),
            'partner_eff': params_array[4:4+n_partners],
            'season_eff': params_array[4+n_partners:4+n_partners+n_seasons],
            'industry_eff': params_array[4+n_partners+n_seasons:]
        }
        
        nll = 0.0
        
        # 正则化
        reg = 0.1
        nll += reg * (params['beta_score']**2 + params['beta_age']**2)
        nll += reg * np.sum(params['partner_eff']**2)
        nll += reg * np.sum(params['season_eff']**2)
        nll += reg * np.sum(params['industry_eff']**2)
        
        for week_data in train_data:
            contestants = week_data['contestants']
            eliminated_names = week_data['eliminated_names']
            method = week_data['method']
            
            if len(contestants) < 2:
                continue
            
            # 计算log投票
            log_votes = self._compute_log_votes(contestants, params)
            
            # 计算评分
            scores = np.array([c['score'] for c in contestants])
            
            # 计算综合淘汰得分
            elim_scores = self._compute_combined_score(scores, log_votes, method)
            
            # 计算淘汰概率
            elim_probs = self._compute_elimination_prob(elim_scores)
            
            # 累加淘汰者的负对数概率
            for i, c in enumerate(contestants):
                if c['name'] in eliminated_names:
                    prob = max(elim_probs[i], 1e-10)
                    nll -= np.log(prob)
        
        return nll
    
    def _optimize_parameters(self, train_data: List[Dict]):
        """优化参数"""
        print("  正在优化参数...")
        
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        # 初始参数
        n_params = 4 + n_partners + n_seasons + n_industries
        x0 = np.zeros(n_params)
        x0[0] = 10.0   # beta_0
        x0[1] = 1.0    # beta_score（正数：高分→高票）
        x0[2] = 0.0    # beta_age
        x0[3] = 0.5    # sigma
        
        # 边界
        bounds = [(5, 15)]              # beta_0
        bounds += [(0.1, 5.0)]          # beta_score（强制为正）
        bounds += [(-1, 1)]             # beta_age
        bounds += [(0.1, 2.0)]          # sigma
        bounds += [(-2, 2)] * n_partners
        bounds += [(-1, 1)] * n_seasons
        bounds += [(-2, 2)] * n_industries
        
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
            print(f"  优化成功! 损失: {result.fun:.2f}")
        else:
            print(f"  优化警告: {result.message}")
        
        # 提取参数
        params = result.x
        self.beta_0 = params[0]
        self.beta_score = params[1]
        self.beta_age = params[2]
        self.sigma = params[3]
        
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
        print(f"    β₀ = {self.beta_0:.3f} (基准投票 ≈ {np.exp(self.beta_0):.0f})")
        print(f"    β_score = {self.beta_score:.3f} (评分效应)")
        print(f"    β_age = {self.beta_age:.3f} (年龄效应)")
        
        # Top舞伴效应
        if self.partner_effects:
            sorted_partners = sorted(self.partner_effects.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  舞伴效应 Top 5:")
            for p, e in sorted_partners[:5]:
                print(f"    {p}: {e:+.3f}")
        
        # Top行业效应
        if self.industry_effects:
            sorted_industries = sorted(self.industry_effects.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  行业效应 Top 5:")
            for ind, e in sorted_industries[:5]:
                print(f"    {ind}: {e:+.3f}")
    
    def _estimate_votes(self, weekly_data: pd.DataFrame):
        """估计所有选手的投票"""
        print("\n  估计投票...")
        
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            for _, row in group.iterrows():
                score_norm = (row['total_score'] - self.score_mean) / self.score_std
                
                age_val = row[self.age_col] if self.age_col else 30
                age_norm = (age_val - self.age_mean) / self.age_std if (self.age_col and pd.notna(age_val)) else 0
                
                # 计算log投票
                log_vote = self.beta_0 + self.beta_score * score_norm + self.beta_age * age_norm
                
                partner = row.get(self.partner_col, 'Unknown') if self.partner_col in row.index else 'Unknown'
                if partner in self.partner_effects:
                    log_vote += self.partner_effects[partner]
                
                if season in self.season_effects:
                    log_vote += self.season_effects[season]
                
                industry = row.get(self.industry_col, 'Unknown') if self.industry_col in row.index else 'Unknown'
                if industry in self.industry_effects:
                    log_vote += self.industry_effects[industry]
                
                votes = np.exp(log_vote)
                
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': row[self.name_col],
                    'total_score': row['total_score'],
                    'log_votes': log_vote,
                    'estimated_votes': votes,
                    'vote_std': votes * self.sigma,
                    'vote_ci_low': np.exp(log_vote - 1.96 * self.sigma),
                    'vote_ci_high': np.exp(log_vote + 1.96 * self.sigma)
                })
        
        self.results_df = pd.DataFrame(results)
        print(f"  完成 {len(results)} 条估计")
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰并计算准确率"""
        if not self.is_fitted:
            raise ValueError("请先调用fit()")
        
        correct = 0
        in_bottom_2 = 0
        total = 0
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            actual_eliminated = elim['eliminated_name'].tolist()
            contestants = group[self.name_col].tolist()
            valid_eliminated = [n for n in actual_eliminated if n in contestants]
            
            if len(valid_eliminated) == 0:
                continue
            
            # 获取该周数据
            week_results = self.results_df[
                (self.results_df['season'] == season) & 
                (self.results_df['week'] == week)
            ]
            
            if len(week_results) == 0:
                continue
            
            # 计算综合淘汰得分
            scores = week_results['total_score'].values
            log_votes = week_results['log_votes'].values
            names = week_results['celebrity'].values
            
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            elim_scores = self._compute_combined_score(scores, log_votes, method)
            
            # 预测（elim_score最高的被淘汰）
            n_elim = len(valid_eliminated)
            pred_indices = np.argsort(elim_scores)[-n_elim:][::-1]
            predicted = [names[i] for i in pred_indices]
            
            # 检查准确率
            is_correct = set(predicted) == set(valid_eliminated)
            
            # 底2
            bottom_2_indices = np.argsort(elim_scores)[-max(2, n_elim):][::-1]
            bottom_2 = [names[i] for i in bottom_2_indices]
            all_in_bottom = all(n in bottom_2 for n in valid_eliminated)
            
            if is_correct:
                correct += 1
            if all_in_bottom:
                in_bottom_2 += 1
            total += 1
            
            results.append({
                'season': season,
                'week': week,
                'actual': valid_eliminated,
                'predicted': predicted,
                'is_correct': is_correct,
                'in_bottom_2': all_in_bottom
            })
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = in_bottom_2 / total if total > 0 else 0
        
        print(f"\n============================================================")
        print(f"淘汰预测结果")
        print(f"============================================================")
        print(f"总周次数: {total}")
        print(f"正确预测数: {correct}")
        print(f"淘汰预测准确率: {accuracy:.2%}")
        print(f"底2准确率: {bottom_accuracy:.2%}")
        print(f"============================================================")
        
        return {
            'accuracy': accuracy,
            'bottom_accuracy': bottom_accuracy,
            'total': total,
            'correct': correct,
            'results': results
        }
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计"""
        return self.results_df
    
    def get_samples_dict(self) -> Dict:
        """返回样本字典（用于不确定性分析）"""
        samples_dict = {}
        for _, row in self.results_df.iterrows():
            key = (row['season'], row['week'], row['celebrity'])
            log_vote = row['log_votes']
            samples = np.exp(np.random.normal(log_vote, self.sigma, 100))
            samples_dict[key] = samples
        return samples_dict
