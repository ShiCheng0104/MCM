"""
贝叶斯淘汰概率模型
使用PyMC进行完整贝叶斯推断，基于淘汰结果学习参数，最终推断观众投票数

核心思路：
1. 投票模型：log(V_i) = β₀ + β_score × score + β_age × age + random_effects + ε
2. 淘汰模型：根据赛季使用排名法或百分比法计算综合得分，最低者被淘汰
3. 似然：最大化实际淘汰者被淘汰的概率
4. 使用全部数据训练，输出每个选手的投票估计
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from scipy.special import softmax
from scipy.stats import rankdata
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC不可用，将使用MLE方法")


# 赛季投票方法分类
SEASONS_RANK_METHOD = list(range(1, 3)) + list(range(28, 35))  # 排名法赛季
SEASONS_PERCENT_METHOD = list(range(3, 28))  # 百分比法赛季


class BayesianEliminationModel:
    """
    贝叶斯淘汰概率模型
    
    模型结构：
    1. 投票模型：log(V_i) = β₀ + β_score × S_i + β_age × Age_i + α_partner + γ_season + δ_industry + ε
    2. 淘汰概率：根据综合得分计算（排名法或百分比法）
    3. 使用贝叶斯推断学习所有参数
    """
    
    def __init__(self, 
                 n_samples: int = 2000,
                 n_tune: int = 2000,
                 n_chains: int = 4,
                 target_accept: float = 0.95,
                 random_seed: int = 42):
        """
        初始化模型
        
        Args:
            n_samples: MCMC采样数量
            n_tune: 调优步数
            n_chains: 马尔可夫链数量
            target_accept: 目标接受率
            random_seed: 随机种子
        """
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 模型参数（后验均值）
        self.beta_0 = 10.0        # log投票基准
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
        self.results_df = None
        self.trace = None
        self.is_fitted = False
    
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """
        使用全部数据训练模型
        
        Args:
            weekly_data: 周级别数据
            elimination_info: 淘汰信息
        """
        print("正在训练贝叶斯淘汰概率模型...")
        
        # 准备训练数据
        train_data = self._prepare_training_data(weekly_data, elimination_info)
        print(f"  准备了 {len(train_data)} 个周次的训练数据")
        
        # 编码分类变量
        self._encode_categorical(weekly_data)
        
        # 贝叶斯推断或MLE
        if PYMC_AVAILABLE:
            self._fit_bayesian(train_data, weekly_data)
        else:
            self._fit_mle(train_data, weekly_data)
        
        # 估计所有选手的投票
        self._estimate_all_votes(weekly_data)
        
        self.is_fitted = True
        print("模型训练完成!")
    
    def _prepare_training_data(self, weekly_data: pd.DataFrame, 
                               elimination_info: pd.DataFrame) -> List[Dict]:
        """准备训练数据"""
        # 标准化评分和年龄
        self.score_mean = weekly_data['total_score'].mean()
        self.score_std = weekly_data['total_score'].std()
        
        if 'celebrity_age' in weekly_data.columns:
            age_col = 'celebrity_age'
        elif 'age' in weekly_data.columns:
            age_col = 'age'
        else:
            age_col = None
        
        if age_col:
            self.age_mean = weekly_data[age_col].mean()
            self.age_std = weekly_data[age_col].std()
        
        # 确定选手名字列
        if 'celebrity_name' in weekly_data.columns:
            name_col = 'celebrity_name'
        else:
            name_col = 'celebrity'
        
        # 确定舞伴和行业列
        partner_col = 'ballroom_partner' if 'ballroom_partner' in weekly_data.columns else 'partner'
        industry_col = 'celebrity_industry' if 'celebrity_industry' in weekly_data.columns else 'industry'
        
        self.name_col = name_col
        self.age_col = age_col
        self.partner_col = partner_col
        self.industry_col = industry_col
        
        train_data = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            # 找到该周的淘汰者
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            eliminated_names = elim['eliminated_name'].tolist()
            contestants = group[name_col].tolist()
            
            valid_eliminated = [name for name in eliminated_names if name in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            # 获取选手数据
            contestants_data = []
            for _, row in group.iterrows():
                age_val = row[age_col] if age_col else 30
                contestants_data.append({
                    'name': row[name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'age': age_val if pd.notna(age_val) else 30,
                    'age_norm': (age_val - self.age_mean) / self.age_std if (age_col and pd.notna(age_val)) else 0,
                    'partner': row.get(partner_col, 'Unknown'),
                    'industry': row.get(industry_col, 'Unknown'),
                    'season': season,
                    'is_eliminated': row[name_col] in valid_eliminated
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
        partner_col = self.partner_col
        industry_col = self.industry_col
        
        if partner_col in weekly_data.columns:
            partners = weekly_data[partner_col].dropna().unique()
            self.partner_to_idx = {p: i for i, p in enumerate(partners)}
        
        seasons = weekly_data['season'].unique()
        self.season_to_idx = {s: i for i, s in enumerate(seasons)}
        
        if industry_col in weekly_data.columns:
            industries = weekly_data[industry_col].dropna().unique()
            self.industry_to_idx = {ind: i for i, ind in enumerate(industries)}
        else:
            self.industry_to_idx = {'Unknown': 0}
    
    def _fit_bayesian(self, train_data: List[Dict], weekly_data: pd.DataFrame):
        """使用PyMC进行贝叶斯推断"""
        print("  使用PyMC进行贝叶斯推断...")
        
        # 为每个选手-周次创建数据
        all_data = []
        for week_data in train_data:
            for c in week_data['contestants']:
                all_data.append({
                    'score_norm': c['score_norm'],
                    'age_norm': c['age_norm'],
                    'partner': c['partner'],
                    'season': c['season'],
                    'industry': c['industry'],
                    'is_eliminated': c['is_eliminated']
                })
        
        df = pd.DataFrame(all_data)
        
        # 创建索引
        partner_idx = df['partner'].map(lambda x: self.partner_to_idx.get(x, 0)).values
        season_idx = df['season'].map(lambda x: self.season_to_idx.get(x, 0)).values
        industry_idx = df['industry'].map(lambda x: self.industry_to_idx.get(x, 0)).values
        
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        with pm.Model() as model:
            # 超先验
            sigma_partner = pm.HalfNormal('sigma_partner', sigma=0.5)
            sigma_season = pm.HalfNormal('sigma_season', sigma=0.3)
            sigma_industry = pm.HalfNormal('sigma_industry', sigma=0.5)
            
            # 固定效应
            beta_0 = pm.Normal('beta_0', mu=10, sigma=2)
            beta_score = pm.TruncatedNormal('beta_score', mu=0.5, sigma=0.5, lower=0)
            beta_age = pm.Normal('beta_age', mu=0, sigma=0.1)
            
            # 随机效应（非中心化）
            if n_partners > 0:
                alpha_partner_raw = pm.Normal('alpha_partner_raw', mu=0, sigma=1, shape=n_partners)
                alpha_partner = pm.Deterministic('alpha_partner', sigma_partner * alpha_partner_raw)
            else:
                alpha_partner = 0
            
            if n_seasons > 0:
                gamma_season_raw = pm.Normal('gamma_season_raw', mu=0, sigma=1, shape=n_seasons)
                gamma_season = pm.Deterministic('gamma_season', sigma_season * gamma_season_raw)
            else:
                gamma_season = 0
            
            if n_industries > 0:
                delta_industry_raw = pm.Normal('delta_industry_raw', mu=0, sigma=1, shape=n_industries)
                delta_industry = pm.Deterministic('delta_industry', sigma_industry * delta_industry_raw)
            else:
                delta_industry = 0
            
            # 残差标准差
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # 线性预测器：log投票
            mu = beta_0 + beta_score * df['score_norm'].values + beta_age * df['age_norm'].values
            
            if n_partners > 0:
                mu = mu + alpha_partner[partner_idx]
            if n_seasons > 0:
                mu = mu + gamma_season[season_idx]
            if n_industries > 0:
                mu = mu + delta_industry[industry_idx]
            
            # 使用淘汰结果作为观测
            # 淘汰者的log投票应该较低，使用潜变量模型
            # 这里我们用一个代理：淘汰者的mu应该较低
            # 使用伯努利似然：P(淘汰) = logistic(-mu * scale)
            scale = pm.HalfNormal('scale', sigma=1)
            p_elim = pm.math.sigmoid(-mu * scale)
            
            # 似然
            y_obs = pm.Bernoulli('y_obs', p=p_elim, observed=df['is_eliminated'].values)
            
            # MCMC采样
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True
            )
        
        # 提取后验均值
        self._extract_posterior_stats()
        
        # 打印诊断
        print("\n  MCMC诊断:")
        summary = az.summary(self.trace, var_names=['beta_0', 'beta_score', 'beta_age', 'sigma', 'scale'])
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
    
    def _fit_mle(self, train_data: List[Dict], weekly_data: pd.DataFrame):
        """使用MLE作为备选"""
        from scipy.optimize import minimize
        
        print("  使用MLE优化...")
        
        n_partners = len(self.partner_to_idx)
        n_seasons = len(self.season_to_idx)
        n_industries = len(self.industry_to_idx)
        
        def neg_log_likelihood(params):
            beta_0 = params[0]
            beta_score = params[1]
            beta_age = params[2]
            sigma = max(params[3], 0.1)
            
            partner_eff = params[4:4+n_partners] if n_partners > 0 else []
            season_eff = params[4+n_partners:4+n_partners+n_seasons] if n_seasons > 0 else []
            industry_eff = params[4+n_partners+n_seasons:] if n_industries > 0 else []
            
            nll = 0.0
            
            for week_data in train_data:
                contestants = week_data['contestants']
                eliminated_names = week_data['eliminated_names']
                
                if len(contestants) < 2:
                    continue
                
                # 计算log投票
                log_votes = []
                for c in contestants:
                    lv = beta_0 + beta_score * c['score_norm'] + beta_age * c['age_norm']
                    
                    if c['partner'] in self.partner_to_idx and n_partners > 0:
                        lv += partner_eff[self.partner_to_idx[c['partner']]]
                    if c['season'] in self.season_to_idx and n_seasons > 0:
                        lv += season_eff[self.season_to_idx[c['season']]]
                    if c.get('industry', 'Unknown') in self.industry_to_idx and n_industries > 0:
                        lv += industry_eff[self.industry_to_idx[c.get('industry', 'Unknown')]]
                    
                    log_votes.append(lv)
                
                log_votes = np.array(log_votes)
                
                # 淘汰概率：投票越低，淘汰概率越高
                elim_probs = softmax(-log_votes)
                
                for i, c in enumerate(contestants):
                    if c['name'] in eliminated_names:
                        prob = max(elim_probs[i], 1e-10)
                        nll -= np.log(prob)
            
            # 正则化
            reg = 0.1
            nll += reg * (beta_score**2 + beta_age**2)
            if n_partners > 0:
                nll += reg * np.sum(np.array(partner_eff)**2)
            if n_seasons > 0:
                nll += reg * np.sum(np.array(season_eff)**2)
            if n_industries > 0:
                nll += reg * np.sum(np.array(industry_eff)**2)
            
            return nll
        
        # 初始化
        n_params = 4 + n_partners + n_seasons + n_industries
        x0 = np.zeros(n_params)
        x0[0] = 10.0  # beta_0
        x0[1] = 0.5   # beta_score
        x0[3] = 0.5   # sigma
        
        # 优化
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                         options={'maxiter': 1000})
        
        if result.success:
            print(f"  MLE优化成功! 损失: {result.fun:.4f}")
        else:
            print(f"  MLE优化警告: {result.message}")
        
        # 提取参数
        params = result.x
        self.beta_0 = params[0]
        self.beta_score = params[1]
        self.beta_age = params[2]
        self.sigma = max(params[3], 0.1)
        
        idx = 4
        if n_partners > 0:
            for partner, i in self.partner_to_idx.items():
                self.partner_effects[partner] = params[idx + i]
            idx += n_partners
        
        if n_seasons > 0:
            for season, i in self.season_to_idx.items():
                self.season_effects[season] = params[idx + i]
            idx += n_seasons
        
        if n_industries > 0:
            for industry, i in self.industry_to_idx.items():
                self.industry_effects[industry] = params[idx + i]
        
        print(f"\n  模型参数:")
        print(f"    β₀ = {self.beta_0:.3f} (基准log投票 ≈ {np.exp(self.beta_0):.0f}票)")
        print(f"    β_score = {self.beta_score:.3f}")
        print(f"    β_age = {self.beta_age:.3f}")
        print(f"    σ = {self.sigma:.3f}")
    
    def _extract_posterior_stats(self):
        """从PyMC trace提取后验统计量"""
        posterior = self.trace.posterior
        
        self.beta_0 = float(posterior['beta_0'].mean())
        self.beta_score = float(posterior['beta_score'].mean())
        self.beta_age = float(posterior['beta_age'].mean())
        self.sigma = float(posterior['sigma'].mean())
        
        # 随机效应
        if 'alpha_partner' in posterior:
            alpha = posterior['alpha_partner'].mean(dim=['chain', 'draw']).values
            for partner, i in self.partner_to_idx.items():
                self.partner_effects[partner] = alpha[i]
        
        if 'gamma_season' in posterior:
            gamma = posterior['gamma_season'].mean(dim=['chain', 'draw']).values
            for season, i in self.season_to_idx.items():
                self.season_effects[season] = gamma[i]
        
        if 'delta_industry' in posterior:
            delta = posterior['delta_industry'].mean(dim=['chain', 'draw']).values
            for industry, i in self.industry_to_idx.items():
                self.industry_effects[industry] = delta[i]
        
        print(f"\n  后验均值:")
        print(f"    β₀ = {self.beta_0:.3f} (基准log投票 ≈ {np.exp(self.beta_0):.0f}票)")
        print(f"    β_score = {self.beta_score:.3f} (评分效应)")
        print(f"    β_age = {self.beta_age:.3f} (年龄效应)")
        print(f"    σ = {self.sigma:.3f}")
    
    def _estimate_all_votes(self, weekly_data: pd.DataFrame):
        """估计所有选手的投票数量"""
        print("\n  正在估计所有选手的投票...")
        
        name_col = self.name_col
        age_col = self.age_col
        partner_col = self.partner_col
        industry_col = self.industry_col
        
        results = []
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            week_scores = []
            week_log_votes = []
            week_names = []
            
            for _, row in group.iterrows():
                # 标准化特征
                score = row['total_score']
                score_norm = (score - self.score_mean) / self.score_std
                
                age_val = row[age_col] if age_col else 30
                age_norm = (age_val - self.age_mean) / self.age_std if (age_col and pd.notna(age_val)) else 0
                
                # 计算log投票
                log_vote = self.beta_0 + self.beta_score * score_norm + self.beta_age * age_norm
                
                # 添加随机效应
                partner = row.get(partner_col, 'Unknown') if partner_col else 'Unknown'
                if partner in self.partner_effects:
                    log_vote += self.partner_effects[partner]
                
                if season in self.season_effects:
                    log_vote += self.season_effects[season]
                
                industry = row.get(industry_col, 'Unknown') if industry_col else 'Unknown'
                if industry in self.industry_effects:
                    log_vote += self.industry_effects[industry]
                
                # 转换为实际投票数
                votes = np.exp(log_vote)
                
                # 生成样本
                vote_samples = np.exp(np.random.normal(log_vote, self.sigma, 100))
                
                week_scores.append(score)
                week_log_votes.append(log_vote)
                week_names.append(row[name_col])
                
                results.append({
                    'season': season,
                    'week': week,
                    'celebrity': row[name_col],
                    'total_score': score,
                    'log_votes': log_vote,
                    'estimated_votes': votes,
                    'vote_std': np.std(vote_samples),
                    'vote_ci_low': np.percentile(vote_samples, 2.5),
                    'vote_ci_high': np.percentile(vote_samples, 97.5)
                })
        
        self.results_df = pd.DataFrame(results)
        print(f"  完成 {len(results)} 条投票估计")
    
    def predict_elimination(self, weekly_data: pd.DataFrame, 
                           elimination_info: pd.DataFrame) -> Dict:
        """预测淘汰结果并计算准确率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        correct = 0
        in_bottom_n = 0
        total = 0
        results_list = []
        
        name_col = self.name_col
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            elim = elimination_info[
                (elimination_info['season'] == season) & 
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            actual_eliminated = elim['eliminated_name'].tolist()
            contestants = group[name_col].tolist()
            
            valid_eliminated = [n for n in actual_eliminated if n in contestants]
            if len(valid_eliminated) == 0:
                continue
            
            # 获取该周的投票估计
            week_results = self.results_df[
                (self.results_df['season'] == season) & 
                (self.results_df['week'] == week)
            ].copy()
            
            if len(week_results) == 0:
                continue
            
            # 根据赛季确定方法
            method = 'rank' if season in SEASONS_RANK_METHOD else 'percent'
            
            # 计算综合得分
            scores = week_results['total_score'].values
            votes = week_results['estimated_votes'].values
            names = week_results['celebrity'].tolist()
            
            if method == 'rank':
                # 排名法：评委排名 + 观众排名
                score_ranks = rankdata(-scores)  # 高分 = 低排名
                vote_ranks = rankdata(-votes)    # 高票 = 低排名
                combined = score_ranks + vote_ranks  # 综合排名（越高越差）
                # 预测淘汰者：综合排名最高的
                pred_indices = np.argsort(combined)[-len(valid_eliminated):]
            else:
                # 百分比法
                score_pct = scores / scores.sum() if scores.sum() > 0 else scores
                vote_pct = votes / votes.sum() if votes.sum() > 0 else votes
                combined = (score_pct + vote_pct) / 2  # 综合百分比
                # 预测淘汰者：综合百分比最低的
                pred_indices = np.argsort(combined)[:len(valid_eliminated)]
            
            pred_names = [names[i] for i in pred_indices]
            
            # 检查准确率
            is_correct = set(pred_names) == set(valid_eliminated)
            
            # 底N检查
            n = max(2, len(valid_eliminated))
            if method == 'rank':
                bottom_n_indices = np.argsort(combined)[-n:]
            else:
                bottom_n_indices = np.argsort(combined)[:n]
            bottom_n_names = [names[i] for i in bottom_n_indices]
            all_in_bottom = all(n in bottom_n_names for n in valid_eliminated)
            
            if is_correct:
                correct += 1
            if all_in_bottom:
                in_bottom_n += 1
            total += 1
            
            results_list.append({
                'season': season,
                'week': week,
                'method': method,
                'actual': valid_eliminated,
                'predicted': pred_names,
                'is_correct': is_correct,
                'in_bottom_n': all_in_bottom
            })
        
        accuracy = correct / total if total > 0 else 0
        bottom_accuracy = in_bottom_n / total if total > 0 else 0
        
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
            'correct': correct,
            'results': results_list
        }
    
    def get_vote_estimates(self) -> pd.DataFrame:
        """返回投票估计结果"""
        return self.results_df
    
    def get_estimates_dict(self) -> Dict:
        """
        返回consistency_check需要的格式
        {(season, week): {'names': [...], 'scores': [...], 'votes': [...]}}
        """
        estimates = {}
        
        for (season, week), group in self.results_df.groupby(['season', 'week']):
            estimates[(season, week)] = {
                'names': group['celebrity'].tolist(),
                'scores': group['total_score'].tolist(),
                'votes': group['estimated_votes'].tolist()
            }
        
        return estimates
    
    def get_samples_dict(self) -> Dict:
        """返回用于不确定性分析的样本"""
        samples_dict = {}
        
        for _, row in self.results_df.iterrows():
            key = (row['season'], row['week'], row['celebrity'])
            samples = np.exp(np.random.normal(row['log_votes'], self.sigma, 100))
            samples_dict[key] = samples
        
        return samples_dict
