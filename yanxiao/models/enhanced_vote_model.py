"""
增强版投票估计模型
结合MLE预热和贝叶斯精调，提高预测准确率

核心改进：
1. 强制评分效应为正且足够大
2. 使用更激进的温度参数
3. 先用MLE快速找到好的初始值，再用贝叶斯精调
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
from scipy.special import softmax
from scipy.stats import rankdata
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

# 赛季投票方法分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))
SEASONS_PERCENT_METHOD = list(range(3, 28))


class EnhancedVoteModel:
    """
    增强版投票估计模型
    
    关键改进：
    1. 使用差分进化算法全局优化
    2. 强制评分效应为正
    3. 使用更小的温度参数增强区分度
    4. 可选贝叶斯精调
    """
    
    def __init__(self, random_seed: int = 42, use_bayesian: bool = True):
        self.random_seed = random_seed
        self.use_bayesian = use_bayesian and PYMC_AVAILABLE
        np.random.seed(random_seed)
        
        # 模型参数
        self.beta_0 = 10.0
        self.beta_score = 2.0     # 增大评分效应初始值
        self.beta_age = 0.0
        self.temperature = 0.5   # 减小温度，增强区分度
        self.sigma = 0.5
        
        # 随机效应
        self.partner_effects = {}
        self.industry_effects = {}
        
        # 编码
        self.partner_to_idx = {}
        self.industry_to_idx = {}
        
        # 标准化
        self.score_mean = 0
        self.score_std = 1
        
        # 列名
        self.name_col = 'celebrity_name'
        self.partner_col = 'ballroom_partner'
        self.industry_col = 'celebrity_industry'
        
        # 结果
        self.results_df = None
        self.trace = None
        self.is_fitted = False
        
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """训练模型"""
        print("正在训练增强版投票模型...")
        
        # 检测列名
        self._detect_columns(weekly_data)
        
        # 准备数据
        train_data = self._prepare_data(weekly_data, elimination_info)
        print(f"  有效训练周次: {len(train_data)}")
        
        # 编码
        self._encode_categorical(weekly_data)
        
        # 第一阶段：全局优化找最佳参数
        print("\n  [阶段1] 全局优化...")
        self._global_optimize(train_data)
        
        # 第二阶段：贝叶斯精调（可选）
        if self.use_bayesian:
            print("\n  [阶段2] 贝叶斯精调...")
            self._bayesian_refine(train_data, weekly_data)
        
        # 估计投票
        print("\n  [阶段3] 估计投票...")
        self._estimate_votes(weekly_data)
        
        self.is_fitted = True
        print("\n模型训练完成!")
        
    def _detect_columns(self, weekly_data: pd.DataFrame):
        """检测列名"""
        if 'celebrity_name' in weekly_data.columns:
            self.name_col = 'celebrity_name'
        elif 'celebrity' in weekly_data.columns:
            self.name_col = 'celebrity'
            
        if 'ballroom_partner' in weekly_data.columns:
            self.partner_col = 'ballroom_partner'
        elif 'partner' in weekly_data.columns:
            self.partner_col = 'partner'
            
        if 'celebrity_industry' in weekly_data.columns:
            self.industry_col = 'celebrity_industry'
        elif 'industry' in weekly_data.columns:
            self.industry_col = 'industry'
    
    def _prepare_data(self, weekly_data: pd.DataFrame, 
                      elimination_info: pd.DataFrame) -> List[Dict]:
        """准备训练数据"""
        self.score_mean = weekly_data['total_score'].mean()
        self.score_std = weekly_data['total_score'].std()
        
        train_data = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = elim['eliminated_name'].tolist()
            contestants = list(group[self.name_col])
            
            valid_eliminated = [n for n in eliminated_names if n in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            contestants_data = []
            for _, row in group.iterrows():
                partner = row.get(self.partner_col, 'Unknown')
                industry = row.get(self.industry_col, 'Unknown')
                
                contestants_data.append({
                    'name': row[self.name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'partner': partner if pd.notna(partner) else 'Unknown',
                    'industry': industry if pd.notna(industry) else 'Unknown',
                    'season': season,
                    'is_eliminated': row[self.name_col] in valid_eliminated
                })
            
            train_data.append({
                'season': season,
                'week': week,
                'contestants': contestants_data,
                'eliminated_names': valid_eliminated,
                'n_eliminated': len(valid_eliminated),
                'method': 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            })
        
        return train_data
    
    def _encode_categorical(self, weekly_data: pd.DataFrame):
        """编码分类变量"""
        if self.partner_col in weekly_data.columns:
            partners = weekly_data[self.partner_col].dropna().unique()
            self.partner_to_idx = {p: i for i, p in enumerate(partners)}
        
        if self.industry_col in weekly_data.columns:
            industries = weekly_data[self.industry_col].dropna().unique()
            self.industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    
    def _compute_log_votes(self, contestants: List[Dict], 
                           beta_0: float, beta_score: float,
                           partner_eff: np.ndarray, 
                           industry_eff: np.ndarray) -> np.ndarray:
        """计算log投票"""
        log_votes = []
        
        for c in contestants:
            lv = beta_0 + beta_score * c['score_norm']
            
            if c['partner'] in self.partner_to_idx:
                idx = self.partner_to_idx[c['partner']]
                if idx < len(partner_eff):
                    lv += partner_eff[idx]
            
            if c.get('industry', 'Unknown') in self.industry_to_idx:
                idx = self.industry_to_idx[c['industry']]
                if idx < len(industry_eff):
                    lv += industry_eff[idx]
            
            log_votes.append(lv)
        
        return np.array(log_votes)
    
    def _compute_elimination_score(self, scores: np.ndarray, 
                                   log_votes: np.ndarray,
                                   method: str) -> np.ndarray:
        """计算淘汰得分（越高越可能被淘汰）"""
        n = len(scores)
        
        if method == 'rank':
            score_ranks = rankdata(-scores)  # 高分=低排名
            vote_ranks = rankdata(-log_votes)
            combined_ranks = score_ranks + vote_ranks
            return combined_ranks  # 排名和越大越可能被淘汰
        else:
            score_pct = scores / scores.sum() if scores.sum() > 0 else np.ones(n) / n
            votes = np.exp(log_votes - log_votes.max())  # 数值稳定
            vote_pct = votes / votes.sum()
            combined_pct = 0.5 * score_pct + 0.5 * vote_pct
            return -combined_pct  # 百分比越小越可能被淘汰
    
    def _compute_week_likelihood(self, contestants: List[Dict],
                                 eliminated_names: List[str],
                                 method: str,
                                 beta_0: float, beta_score: float,
                                 partner_eff: np.ndarray,
                                 industry_eff: np.ndarray,
                                 temperature: float) -> float:
        """计算某周的对数似然"""
        scores = np.array([c['score'] for c in contestants])
        log_votes = self._compute_log_votes(
            contestants, beta_0, beta_score, partner_eff, industry_eff
        )
        
        elim_scores = self._compute_elimination_score(scores, log_votes, method)
        elim_probs = softmax(elim_scores / temperature)
        
        # 被淘汰者的概率
        ll = 0.0
        for i, c in enumerate(contestants):
            if c['name'] in eliminated_names:
                ll += np.log(max(elim_probs[i], 1e-10))
        
        return ll
    
    def _objective(self, params: np.ndarray, train_data: List[Dict]) -> float:
        """优化目标：负对数似然 + 正则化"""
        n_partners = len(self.partner_to_idx)
        n_industries = len(self.industry_to_idx)
        
        beta_0 = params[0]
        beta_score = params[1]
        temperature = params[2]
        partner_eff = params[3:3+n_partners]
        industry_eff = params[3+n_partners:3+n_partners+n_industries]
        
        nll = 0.0
        
        # 正则化
        reg = 0.01
        nll += reg * np.sum(partner_eff**2)
        nll += reg * np.sum(industry_eff**2)
        
        for week_data in train_data:
            ll = self._compute_week_likelihood(
                week_data['contestants'],
                week_data['eliminated_names'],
                week_data['method'],
                beta_0, beta_score,
                partner_eff, industry_eff,
                temperature
            )
            nll -= ll
        
        return nll
    
    def _global_optimize(self, train_data: List[Dict]):
        """使用差分进化全局优化"""
        n_partners = len(self.partner_to_idx)
        n_industries = len(self.industry_to_idx)
        n_params = 3 + n_partners + n_industries
        
        # 参数边界
        bounds = [
            (5, 15),      # beta_0
            (0.5, 5.0),   # beta_score（强制为正且足够大）
            (0.1, 2.0),   # temperature
        ]
        bounds += [(-1, 1)] * n_partners
        bounds += [(-1, 1)] * n_industries
        
        # 差分进化（全局优化）
        result = differential_evolution(
            self._objective,
            bounds,
            args=(train_data,),
            seed=self.random_seed,
            maxiter=100,
            polish=True,
            disp=False
        )
        
        # 提取参数
        self.beta_0 = result.x[0]
        self.beta_score = result.x[1]
        self.temperature = result.x[2]
        
        partner_eff = result.x[3:3+n_partners]
        industry_eff = result.x[3+n_partners:]
        
        for partner, idx in self.partner_to_idx.items():
            self.partner_effects[partner] = partner_eff[idx]
        
        for industry, idx in self.industry_to_idx.items():
            self.industry_effects[industry] = industry_eff[idx]
        
        print(f"    优化完成! 损失: {result.fun:.4f}")
        print(f"    β₀ = {self.beta_0:.3f}")
        print(f"    β_score = {self.beta_score:.3f}")
        print(f"    temperature = {self.temperature:.3f}")
        
        # 计算训练准确率
        self._compute_train_accuracy(train_data)
    
    def _compute_train_accuracy(self, train_data: List[Dict]):
        """计算训练集准确率"""
        correct = 0
        total = 0
        
        partner_eff = np.array([self.partner_effects.get(p, 0) 
                                for p in sorted(self.partner_to_idx.keys(), 
                                               key=lambda x: self.partner_to_idx[x])])
        industry_eff = np.array([self.industry_effects.get(i, 0) 
                                 for i in sorted(self.industry_to_idx.keys(), 
                                                key=lambda x: self.industry_to_idx[x])])
        
        for week_data in train_data:
            contestants = week_data['contestants']
            eliminated_names = week_data['eliminated_names']
            method = week_data['method']
            n_elim = len(eliminated_names)
            
            scores = np.array([c['score'] for c in contestants])
            log_votes = self._compute_log_votes(
                contestants, self.beta_0, self.beta_score,
                partner_eff, industry_eff
            )
            
            elim_scores = self._compute_elimination_score(scores, log_votes, method)
            
            # 预测淘汰者（淘汰得分最高的n_elim个）
            pred_indices = np.argsort(elim_scores)[-n_elim:]
            pred_names = [contestants[i]['name'] for i in pred_indices]
            
            if set(pred_names) == set(eliminated_names):
                correct += 1
            total += 1
        
        print(f"    训练准确率: {correct}/{total} = {correct/total:.2%}")
    
    def _bayesian_refine(self, train_data: List[Dict], weekly_data: pd.DataFrame):
        """贝叶斯精调"""
        if not PYMC_AVAILABLE:
            print("    PyMC不可用，跳过贝叶斯精调")
            return
        
        # 准备观测数据
        all_scores = []
        all_is_eliminated = []
        all_partner_idx = []
        all_industry_idx = []
        
        for week_data in train_data:
            for c in week_data['contestants']:
                all_scores.append(c['score_norm'])
                all_is_eliminated.append(1 if c['is_eliminated'] else 0)
                all_partner_idx.append(
                    self.partner_to_idx.get(c['partner'], 0)
                )
                all_industry_idx.append(
                    self.industry_to_idx.get(c.get('industry', 'Unknown'), 0)
                )
        
        scores = np.array(all_scores)
        is_eliminated = np.array(all_is_eliminated)
        partner_idx = np.array(all_partner_idx)
        industry_idx = np.array(all_industry_idx)
        
        n_partners = len(self.partner_to_idx)
        n_industries = len(self.industry_to_idx)
        
        with pm.Model() as model:
            # 先验（以MLE结果为中心）
            beta_0 = pm.Normal('beta_0', mu=self.beta_0, sigma=1)
            beta_score = pm.TruncatedNormal('beta_score', mu=self.beta_score, 
                                            sigma=0.5, lower=0.1)
            
            sigma_partner = pm.HalfNormal('sigma_partner', sigma=0.3)
            sigma_industry = pm.HalfNormal('sigma_industry', sigma=0.3)
            
            # 随机效应
            partner_eff_raw = pm.Normal('partner_eff_raw', mu=0, sigma=1, 
                                        shape=n_partners)
            partner_eff = pm.Deterministic('partner_eff', 
                                           sigma_partner * partner_eff_raw)
            
            industry_eff_raw = pm.Normal('industry_eff_raw', mu=0, sigma=1,
                                         shape=n_industries)
            industry_eff = pm.Deterministic('industry_eff',
                                            sigma_industry * industry_eff_raw)
            
            # 线性预测器
            mu = (beta_0 + 
                  beta_score * scores +
                  partner_eff[partner_idx] +
                  industry_eff[industry_idx])
            
            # 淘汰概率（logistic回归）
            # 高投票 → 低mu → 低淘汰概率
            p_elim = pm.math.invlogit(-mu * 0.5)
            
            # 似然
            y = pm.Bernoulli('y', p=p_elim, observed=is_eliminated)
            
            # MCMC采样
            self.trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=2,
                target_accept=0.9,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True
            )
        
        # 提取后验均值
        posterior = self.trace.posterior
        self.beta_0 = float(posterior['beta_0'].mean())
        self.beta_score = float(posterior['beta_score'].mean())
        
        partner_eff_post = posterior['partner_eff'].mean(dim=['chain', 'draw']).values
        industry_eff_post = posterior['industry_eff'].mean(dim=['chain', 'draw']).values
        
        for partner, idx in self.partner_to_idx.items():
            self.partner_effects[partner] = partner_eff_post[idx]
        
        for industry, idx in self.industry_to_idx.items():
            self.industry_effects[industry] = industry_eff_post[idx]
        
        # 打印诊断
        print("\n    MCMC诊断:")
        summary = az.summary(self.trace, var_names=['beta_0', 'beta_score'])
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
        
        # 重新计算训练准确率
        self._compute_train_accuracy(train_data)
    
    def _estimate_votes(self, weekly_data: pd.DataFrame):
        """估计投票"""
        results = []
        
        partner_eff = np.array([self.partner_effects.get(p, 0) 
                                for p in sorted(self.partner_to_idx.keys(), 
                                               key=lambda x: self.partner_to_idx[x])])
        industry_eff = np.array([self.industry_effects.get(i, 0) 
                                 for i in sorted(self.industry_to_idx.keys(), 
                                                key=lambda x: self.industry_to_idx[x])])
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            contestants = []
            for _, row in group.iterrows():
                partner = row.get(self.partner_col, 'Unknown')
                industry = row.get(self.industry_col, 'Unknown')
                
                contestants.append({
                    'name': row[self.name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'partner': partner if pd.notna(partner) else 'Unknown',
                    'industry': industry if pd.notna(industry) else 'Unknown'
                })
            
            log_votes = self._compute_log_votes(
                contestants, self.beta_0, self.beta_score,
                partner_eff, industry_eff
            )
            votes = np.exp(log_votes)
            
            # 生成样本
            for i, c in enumerate(contestants):
                vote_samples = np.exp(
                    np.random.normal(log_votes[i], self.sigma, 100)
                )
                
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': c['name'],
                    'total_score': c['score'],
                    'estimated_votes': votes[i],
                    'log_votes': log_votes[i],
                    'vote_ci_low': np.percentile(vote_samples, 2.5),
                    'vote_ci_high': np.percentile(vote_samples, 97.5)
                })
        
        self.results_df = pd.DataFrame(results)
        print(f"    完成 {len(results)} 条投票估计")
    
    def predict_elimination(self, weekly_data: pd.DataFrame,
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰并计算准确率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        correct = 0
        in_bottom = 0
        total = 0
        
        partner_eff = np.array([self.partner_effects.get(p, 0) 
                                for p in sorted(self.partner_to_idx.keys(), 
                                               key=lambda x: self.partner_to_idx[x])])
        industry_eff = np.array([self.industry_effects.get(i, 0) 
                                 for i in sorted(self.industry_to_idx.keys(), 
                                                key=lambda x: self.industry_to_idx[x])])
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = elim['eliminated_name'].tolist()
            contestants_names = list(group[self.name_col])
            valid_eliminated = [n for n in eliminated_names if n in contestants_names]
            
            if len(valid_eliminated) == 0:
                continue
            
            contestants = []
            for _, row in group.iterrows():
                partner = row.get(self.partner_col, 'Unknown')
                industry = row.get(self.industry_col, 'Unknown')
                
                contestants.append({
                    'name': row[self.name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'partner': partner if pd.notna(partner) else 'Unknown',
                    'industry': industry if pd.notna(industry) else 'Unknown'
                })
            
            scores = np.array([c['score'] for c in contestants])
            log_votes = self._compute_log_votes(
                contestants, self.beta_0, self.beta_score,
                partner_eff, industry_eff
            )
            
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            elim_scores = self._compute_elimination_score(scores, log_votes, method)
            
            n_elim = len(valid_eliminated)
            pred_indices = np.argsort(elim_scores)[-n_elim:]
            pred_names = [contestants[i]['name'] for i in pred_indices]
            
            if set(pred_names) == set(valid_eliminated):
                correct += 1
            
            # 检查是否在底部
            bottom_n = max(2, n_elim)
            bottom_indices = np.argsort(elim_scores)[-bottom_n:]
            bottom_names = [contestants[i]['name'] for i in bottom_indices]
            if all(n in bottom_names for n in valid_eliminated):
                in_bottom += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = in_bottom / total if total > 0 else 0
        
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
        """返回投票估计"""
        return self.results_df
    
    def get_estimates_dict(self) -> Dict:
        """返回字典格式的估计（用于一致性检验）"""
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
        samples = {}
        
        for _, row in self.results_df.iterrows():
            key = (row['season'], row['week'], row['celebrity'])
            samples[key] = np.exp(
                np.random.normal(row['log_votes'], self.sigma, 100)
            )
        
        return samples
