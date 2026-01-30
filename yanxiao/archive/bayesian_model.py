"""
贝叶斯层次模型
使用PyMC进行完整的贝叶斯推断建模观众投票
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError as e:
    PYMC_AVAILABLE = False
    warnings.warn(f"PyMC导入失败: {e}。将使用简化的MCMC采样。建议运行: pip install pymc arviz")
except Exception as e:
    PYMC_AVAILABLE = False
    warnings.warn(f"PyMC加载异常: {e}。将使用简化的MCMC采样。")

from src.utils import normalize_votes


class BayesianVoteModel:
    """
    贝叶斯层次投票估计模型（完整贝叶斯推断）
    
    模型结构：
    log(V_{i,w}) ~ Normal(μ_{i,w}, σ²)
    μ_{i,w} = β₀ + β₁·S_{i,w} + β₂·Age_i + α_partner[j] + γ_season[k] + δ_industry[l]
    
    先验分布：
    β₀ ~ Normal(10, 2)
    β₁ ~ Normal(0, 0.5)
    β₂ ~ Normal(0, 0.1)
    α_partner ~ Normal(0, σ_partner), σ_partner ~ HalfNormal(0.5)
    γ_season ~ Normal(0, σ_season), σ_season ~ HalfNormal(0.3)
    δ_industry ~ Normal(0, σ_industry), σ_industry ~ HalfNormal(0.5)
    σ ~ HalfNormal(1)
    
    使用PyMC进行MCMC采样进行完整的贝叶斯推断
    """
    
    def __init__(self, 
                 n_samples: int = 2000,
                 n_tune: int = 2000,
                 n_chains: int = 4,
                 target_accept: float = 0.95,
                 random_seed: int = 42):
        """
        初始化贝叶斯模型
        
        Args:
            n_samples: MCMC采样数量
            n_tune: 调优步数（增加以改善收敛）
            n_chains: 马尔可夫链数量（增加以获得更可靠的收敛诊断）
            target_accept: 目标接受率（提高以减少divergences）
            random_seed: 随机种子
        """
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 后验参数（拟合后更新）
        self.trace = None
        self.posterior_samples = None
        
        # 模型参数的后验均值
        self.beta_0_mean = 10.0
        self.beta_score_mean = 0.1
        self.beta_age_mean = -0.01
        self.sigma_mean = 0.5
        
        # 随机效应后验
        self.partner_effects = {}
        self.season_effects = {}
        self.industry_effects = {}
        
        # 编码映射
        self.partner_to_idx = {}
        self.season_to_idx = {}
        self.industry_to_idx = {}
        
        # 存储结果
        self.estimates = {}
        self.samples = {}
        self.model = None
        self.is_fitted = False
        
        # 赛季内标准化参数
        self.season_score_stats = {}  # {season: (mean, std)}
        self.season_age_stats = {}    # {season: (mean, std)}
        
    def _encode_categorical(self, data: pd.DataFrame):
        """
        编码分类变量
        
        Args:
            data: 周级别数据
        """
        # 舞伴编码
        partners = data['ballroom_partner'].unique()
        self.partner_to_idx = {p: i for i, p in enumerate(partners)}
        
        # 赛季编码
        seasons = sorted(data['season'].unique())
        self.season_to_idx = {s: i for i, s in enumerate(seasons)}
        
        # 行业编码
        industries = data['celebrity_industry'].dropna().unique()
        self.industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    
    def _prepare_data_for_pymc(self, data: pd.DataFrame) -> Dict:
        """
        准备PyMC模型所需的数据格式
        
        Args:
            data: 周级别数据
        
        Returns:
            格式化的数据字典
        """
        self._encode_categorical(data)
        
        # 计算并存储每个赛季的统计量
        for season in data['season'].unique():
            season_data = data[data['season'] == season]
            score_mean = season_data['total_score'].mean()
            score_std = season_data['total_score'].std()
            self.season_score_stats[season] = (score_mean, max(score_std, 1.0))
            
            age_mean = season_data['celebrity_age'].fillna(35).mean()
            age_std = season_data['celebrity_age'].fillna(35).std()
            self.season_age_stats[season] = (age_mean, max(age_std, 1.0))
        
        # 赛季内标准化得分（每个赛季单独计算均值和标准差）
        scores_normalized = data.groupby('season')['total_score'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        ).values
        
        # 赛季内标准化年龄（使用transform避免FutureWarning）
        ages_filled = data['celebrity_age'].fillna(35)
        ages_normalized = ages_filled.groupby(data['season']).transform(
            lambda x: (x - x.mean()) / max(x.std(), 1.0)
        ).values
        
        # 编码分类变量
        partner_idx = data['ballroom_partner'].map(self.partner_to_idx).values
        season_idx = data['season'].map(self.season_to_idx).values
        industry_idx = data['celebrity_industry'].map(
            lambda x: self.industry_to_idx.get(x, 0) if pd.notna(x) else 0
        ).values
        
        # 全局统计（用于fallback）
        scores = data['total_score'].values
        ages = data['celebrity_age'].fillna(35).values
        
        return {
            'scores': scores_normalized,
            'ages': ages_normalized,
            'partner_idx': partner_idx.astype(int),
            'season_idx': season_idx.astype(int),
            'industry_idx': industry_idx.astype(int),
            'n_partners': len(self.partner_to_idx),
            'n_seasons': len(self.season_to_idx),
            'n_industries': max(len(self.industry_to_idx), 1),
            'n_obs': len(data),
            'scores_mean': scores.mean(),
            'scores_std': max(scores.std(), 1.0),
            'ages_mean': ages.mean(),
            'ages_std': max(ages.std(), 1.0)
        }
    
    def fit(self, weekly_data: pd.DataFrame, elimination_info: pd.DataFrame):
        """
        使用PyMC进行完整的贝叶斯推断拟合模型
        
        Args:
            weekly_data: 周级别数据
            elimination_info: 淘汰信息
        """
        print("正在进行贝叶斯推断...")
        
        # 准备数据
        model_data = self._prepare_data_for_pymc(weekly_data)
        
        if PYMC_AVAILABLE:
            self._fit_with_pymc(model_data, weekly_data)
        else:
            self._fit_with_metropolis_hastings(model_data, weekly_data, elimination_info)
        
        self.is_fitted = True
        print("贝叶斯推断完成!")
    
    def _fit_with_pymc(self, model_data: Dict, weekly_data: pd.DataFrame):
        """
        使用PyMC进行MCMC采样
        
        模型设计：
        1. 预测"投票吸引力"（latent vote appeal）
        2. 投票吸引力 = beta_score * 评分 + beta_age * 年龄 + 随机效应
        3. 使用评分作为观测数据来识别随机效应
        4. beta_score 通过先验知识设定（高分→高票）
        
        Args:
            model_data: 格式化的数据
            weekly_data: 原始周级别数据
        """
        print("  使用PyMC进行MCMC采样...")
        
        # 准备观测数据：使用标准化后的评分
        observed_scores = model_data['scores']
        
        with pm.Model() as self.model:
            # ========== 超先验（控制随机效应的方差）==========
            sigma_partner = pm.HalfNormal('sigma_partner', sigma=0.5)
            sigma_season = pm.HalfNormal('sigma_season', sigma=0.3)
            sigma_industry = pm.HalfNormal('sigma_industry', sigma=0.5)
            
            # ========== 固定效应先验 ==========
            # beta_0: 基准值
            beta_0 = pm.Normal('beta_0', mu=0, sigma=1)
            # beta_score: 评分对投票的影响（正值，高分→高票）
            # 使用信息先验：评分效应应该是正的
            beta_score = pm.TruncatedNormal('beta_score', mu=0.5, sigma=0.3, lower=0.1, upper=2.0)
            # beta_age: 年龄效应
            beta_age = pm.Normal('beta_age', mu=0, sigma=0.1)
            
            # ========== 非中心化随机效应 ==========
            alpha_partner_raw = pm.Normal('alpha_partner_raw', mu=0, sigma=1, 
                                          shape=model_data['n_partners'])
            alpha_partner = pm.Deterministic('alpha_partner', 
                                             sigma_partner * alpha_partner_raw)
            
            gamma_season_raw = pm.Normal('gamma_season_raw', mu=0, sigma=1,
                                         shape=model_data['n_seasons'])
            gamma_season = pm.Deterministic('gamma_season', 
                                            sigma_season * gamma_season_raw)
            
            delta_industry_raw = pm.Normal('delta_industry_raw', mu=0, sigma=1,
                                           shape=model_data['n_industries'])
            delta_industry = pm.Deterministic('delta_industry', 
                                              sigma_industry * delta_industry_raw)
            
            # ========== 残差标准差 ==========
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # ========== 预测评分（用于识别随机效应）==========
            # 评分预测模型：score = beta_0 + random_effects + noise
            mu_score = (beta_0 + 
                        beta_age * model_data['ages'] +
                        alpha_partner[model_data['partner_idx']] +
                        gamma_season[model_data['season_idx']] +
                        delta_industry[model_data['industry_idx']])
            
            # 使用评分作为观测数据约束随机效应
            score_obs = pm.Normal('score_obs', mu=mu_score, sigma=sigma, 
                                  observed=observed_scores)
            
            # ========== MCMC采样 ==========
            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=True
            )
        
        # 提取后验统计
        self._extract_posterior_stats()
        
        # 打印诊断信息
        print("\n  MCMC诊断:")
        summary = az.summary(self.trace, var_names=['beta_0', 'beta_score', 'beta_age', 'sigma', 
                                                     'sigma_partner', 'sigma_season', 'sigma_industry'])
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
    
    def _check_numpyro(self) -> bool:
        """检查numpyro是否可用"""
        try:
            import numpyro
            import jax
            return True
        except ImportError:
            return False
    
    def _fit_with_metropolis_hastings(self, model_data: Dict, 
                                       weekly_data: pd.DataFrame,
                                       elimination_info: pd.DataFrame):
        """
        使用自实现的Metropolis-Hastings采样（当PyMC不可用时）
        
        Args:
            model_data: 格式化的数据
            weekly_data: 周级别数据
            elimination_info: 淘汰信息
        """
        print("  使用Metropolis-Hastings采样（PyMC不可用）...")
        
        n_iter = self.n_samples + self.n_tune
        
        # 参数维度
        n_partners = model_data['n_partners']
        n_seasons = model_data['n_seasons']
        n_industries = model_data['n_industries']
        
        # 初始化参数
        params = {
            'beta_0': 10.0,
            'beta_score': 0.1,
            'beta_age': -0.01,
            'sigma': 0.5,
            'sigma_partner': 0.3,
            'sigma_season': 0.2,
            'sigma_industry': 0.3,
            'alpha_partner': np.zeros(n_partners),
            'gamma_season': np.zeros(n_seasons),
            'delta_industry': np.zeros(n_industries)
        }
        
        # 提案分布标准差
        proposal_sd = {
            'beta_0': 0.1,
            'beta_score': 0.02,
            'beta_age': 0.005,
            'sigma': 0.05,
            'sigma_partner': 0.02,
            'sigma_season': 0.02,
            'sigma_industry': 0.02,
            'alpha_partner': 0.05,
            'gamma_season': 0.03,
            'delta_industry': 0.05
        }
        
        # 存储采样
        samples = {k: [] for k in params.keys()}
        
        def log_prior(p):
            """计算先验对数概率"""
            lp = 0
            lp += stats.norm.logpdf(p['beta_0'], 10, 2)
            lp += stats.norm.logpdf(p['beta_score'], 0, 0.5)
            lp += stats.norm.logpdf(p['beta_age'], 0, 0.1)
            lp += stats.halfnorm.logpdf(p['sigma'], scale=1)
            lp += stats.halfnorm.logpdf(p['sigma_partner'], scale=0.5)
            lp += stats.halfnorm.logpdf(p['sigma_season'], scale=0.3)
            lp += stats.halfnorm.logpdf(p['sigma_industry'], scale=0.5)
            lp += np.sum(stats.norm.logpdf(p['alpha_partner'], 0, p['sigma_partner']))
            lp += np.sum(stats.norm.logpdf(p['gamma_season'], 0, p['sigma_season']))
            lp += np.sum(stats.norm.logpdf(p['delta_industry'], 0, p['sigma_industry']))
            return lp
        
        def log_likelihood(p, data):
            """计算似然对数概率"""
            mu = (p['beta_0'] + 
                  p['beta_score'] * data['scores'] +
                  p['beta_age'] * data['ages'] +
                  p['alpha_partner'][data['partner_idx']] +
                  p['gamma_season'][data['season_idx']] +
                  p['delta_industry'][data['industry_idx']])
            
            # 使用得分的对数作为观测值的代理
            log_obs = np.log(weekly_data['total_score'].values + 1)
            log_obs_normalized = (log_obs - log_obs.mean()) / log_obs.std() * 2 + 10
            
            ll = np.sum(stats.norm.logpdf(log_obs_normalized, mu, p['sigma']))
            return ll
        
        # Metropolis-Hastings采样
        current_lp = log_prior(params) + log_likelihood(params, model_data)
        accepted = 0
        
        for i in range(n_iter):
            # 随机选择一个参数更新
            param_name = np.random.choice(list(params.keys()))
            
            # 提案
            proposed = {k: v.copy() if isinstance(v, np.ndarray) else v 
                       for k, v in params.items()}
            
            if isinstance(params[param_name], np.ndarray):
                idx = np.random.randint(len(params[param_name]))
                proposed[param_name][idx] += np.random.normal(0, proposal_sd[param_name])
            else:
                proposed[param_name] += np.random.normal(0, proposal_sd[param_name])
            
            # 确保正值参数为正
            if param_name in ['sigma', 'sigma_partner', 'sigma_season', 'sigma_industry']:
                if isinstance(proposed[param_name], np.ndarray):
                    proposed[param_name] = np.abs(proposed[param_name])
                else:
                    proposed[param_name] = abs(proposed[param_name])
            
            # 计算接受概率
            proposed_lp = log_prior(proposed) + log_likelihood(proposed, model_data)
            
            if np.log(np.random.random()) < proposed_lp - current_lp:
                params = proposed
                current_lp = proposed_lp
                accepted += 1
            
            # 保存样本（丢弃burn-in）
            if i >= self.n_tune:
                for k, v in params.items():
                    samples[k].append(v.copy() if isinstance(v, np.ndarray) else v)
            
            # 进度提示
            if (i + 1) % 500 == 0:
                print(f"    迭代 {i+1}/{n_iter}, 接受率: {accepted/(i+1):.2%}")
        
        # 转换为numpy数组
        self.posterior_samples = {k: np.array(v) for k, v in samples.items()}
        
        # 提取后验均值
        self._extract_posterior_stats_from_samples()
        
        print(f"\n  采样完成，总接受率: {accepted/n_iter:.2%}")
    
    def _extract_posterior_stats(self):
        """从PyMC trace中提取后验统计量"""
        posterior = self.trace.posterior
        
        self.beta_0_mean = float(posterior['beta_0'].mean())
        self.beta_score_mean = float(posterior['beta_score'].mean())  # 评分对投票的影响
        self.beta_age_mean = float(posterior['beta_age'].mean())
        self.sigma_mean = float(posterior['sigma'].mean())
        
        # 提取随机效应
        alpha_partner = posterior['alpha_partner'].mean(dim=['chain', 'draw']).values
        gamma_season = posterior['gamma_season'].mean(dim=['chain', 'draw']).values
        delta_industry = posterior['delta_industry'].mean(dim=['chain', 'draw']).values
        
        # 映射回原始类别
        idx_to_partner = {v: k for k, v in self.partner_to_idx.items()}
        idx_to_season = {v: k for k, v in self.season_to_idx.items()}
        idx_to_industry = {v: k for k, v in self.industry_to_idx.items()}
        
        self.partner_effects = {idx_to_partner[i]: alpha_partner[i] 
                               for i in range(len(alpha_partner))}
        self.season_effects = {idx_to_season[i]: gamma_season[i] 
                              for i in range(len(gamma_season))}
        self.industry_effects = {idx_to_industry[i]: delta_industry[i] 
                                for i in range(len(delta_industry))}
        
        print(f"\n  后验均值:")
        print(f"    β₀ = {self.beta_0_mean:.3f}")
        print(f"    β_score = {self.beta_score_mean:.3f} (评分→投票效应)")
        print(f"    β_age = {self.beta_age_mean:.3f}")
        print(f"    σ = {self.sigma_mean:.3f}")
    
    def _extract_posterior_stats_from_samples(self):
        """从自采样结果中提取后验统计量"""
        samples = self.posterior_samples
        
        self.beta_0_mean = np.mean(samples['beta_0'])
        self.beta_score_mean = np.mean(samples['beta_score'])
        self.beta_age_mean = np.mean(samples['beta_age'])
        self.sigma_mean = np.mean(samples['sigma'])
        
        alpha_partner = np.mean(samples['alpha_partner'], axis=0)
        gamma_season = np.mean(samples['gamma_season'], axis=0)
        delta_industry = np.mean(samples['delta_industry'], axis=0)
        
        idx_to_partner = {v: k for k, v in self.partner_to_idx.items()}
        idx_to_season = {v: k for k, v in self.season_to_idx.items()}
        idx_to_industry = {v: k for k, v in self.industry_to_idx.items()}
        
        self.partner_effects = {idx_to_partner[i]: alpha_partner[i] 
                               for i in range(len(alpha_partner))}
        self.season_effects = {idx_to_season[i]: gamma_season[i] 
                              for i in range(len(gamma_season))}
        self.industry_effects = {idx_to_industry.get(i, f'unknown_{i}'): delta_industry[i] 
                                for i in range(len(delta_industry))}
        
        print(f"\n  后验均值:")
        print(f"    β₀ = {self.beta_0_mean:.3f}")
        print(f"    β_score = {self.beta_score_mean:.3f}")
        print(f"    β_age = {self.beta_age_mean:.3f}")
        print(f"    σ = {self.sigma_mean:.3f}")
    
    def compute_expected_log_votes(self,
                                   score: float,
                                   age: float,
                                   partner: str,
                                   season: int,
                                   industry: Optional[str] = None,
                                   scores_mean: float = 25.0,
                                   scores_std: float = 5.0,
                                   ages_mean: float = 35.0,
                                   ages_std: float = 10.0) -> float:
        """
        计算期望的对数投票数
        
        Args:
            score: 评委总分
            age: 选手年龄
            partner: 舞伴姓名
            season: 赛季编号
            industry: 行业类别
            scores_mean: 得分均值（用于标准化）
            scores_std: 得分标准差
            ages_mean: 年龄均值
            ages_std: 年龄标准差
        
        Returns:
            期望对数投票
        """
        # 标准化输入
        score_norm = (score - scores_mean) / scores_std if scores_std > 0 else 0
        age_norm = (age - ages_mean) / ages_std if ages_std > 0 else 0
        
        mu = self.beta_0_mean
        mu += self.beta_score_mean * score_norm
        mu += self.beta_age_mean * age_norm
        
        # 添加随机效应
        if partner in self.partner_effects:
            mu += self.partner_effects[partner]
        
        if season in self.season_effects:
            mu += self.season_effects[season]
        
        if industry and industry in self.industry_effects:
            mu += self.industry_effects[industry]
        
        return mu
    
    def sample_votes_posterior(self,
                               contestants: pd.DataFrame,
                               n_samples: Optional[int] = None) -> np.ndarray:
        """
        从后验预测分布采样投票
        
        Args:
            contestants: 选手数据
            n_samples: 采样数量
        
        Returns:
            样本数组 (n_samples, n_contestants)
        """
        if n_samples is None:
            n_samples = min(self.n_samples, 1000)
        
        n_contestants = len(contestants)
        samples = np.zeros((n_samples, n_contestants))
        
        # 获取赛季内标准化的统计量
        season = contestants['season'].iloc[0]  # 同一周的选手都是同一赛季
        if season in self.season_score_stats:
            scores_mean, scores_std = self.season_score_stats[season]
            ages_mean, ages_std = self.season_age_stats[season]
        else:
            # Fallback: 使用当前数据的统计量
            all_scores = contestants['total_score'].values
            all_ages = contestants['celebrity_age'].fillna(35).values
            scores_mean, scores_std = all_scores.mean(), max(all_scores.std(), 1)
            ages_mean, ages_std = all_ages.mean(), max(all_ages.std(), 1)
        
        if self.posterior_samples is not None:
            # 使用后验样本
            n_posterior = len(self.posterior_samples['beta_0'])
            sample_indices = np.random.choice(n_posterior, n_samples, replace=True)
            
            for s_idx, p_idx in enumerate(sample_indices):
                beta_0 = self.posterior_samples['beta_0'][p_idx]
                beta_score = self.posterior_samples['beta_score'][p_idx]
                beta_age = self.posterior_samples['beta_age'][p_idx]
                sigma = self.posterior_samples['sigma'][p_idx]
                alpha_partner = self.posterior_samples['alpha_partner'][p_idx]
                gamma_season = self.posterior_samples['gamma_season'][p_idx]
                delta_industry = self.posterior_samples['delta_industry'][p_idx]
                
                for i, (_, row) in enumerate(contestants.iterrows()):
                    score_norm = (row['total_score'] - scores_mean) / scores_std
                    age = row.get('celebrity_age', 35)
                    if pd.isna(age):
                        age = 35
                    age_norm = (age - ages_mean) / ages_std
                    
                    partner_idx = self.partner_to_idx.get(row['ballroom_partner'], 0)
                    season_idx = self.season_to_idx.get(row['season'], 0)
                    industry_idx = self.industry_to_idx.get(row.get('celebrity_industry'), 0)
                    
                    mu = (beta_0 + 
                          beta_score * score_norm +
                          beta_age * age_norm +
                          alpha_partner[partner_idx] +
                          gamma_season[season_idx] +
                          delta_industry[industry_idx])
                    
                    log_vote = np.random.normal(mu, sigma)
                    samples[s_idx, i] = np.exp(log_vote)
        else:
            # 使用后验均值 + 不确定性
            for i, (_, row) in enumerate(contestants.iterrows()):
                mu = self.compute_expected_log_votes(
                    score=row['total_score'],
                    age=row.get('celebrity_age', 35) if pd.notna(row.get('celebrity_age')) else 35,
                    partner=row['ballroom_partner'],
                    season=row['season'],
                    industry=row.get('celebrity_industry', None),
                    scores_mean=scores_mean,
                    scores_std=scores_std,
                    ages_mean=ages_mean,
                    ages_std=ages_std
                )
                
                log_votes = np.random.normal(mu, self.sigma_mean, n_samples)
                samples[:, i] = np.exp(log_votes)
        
        # 归一化每个样本
        for j in range(n_samples):
            samples[j] = normalize_votes(samples[j], total_votes=1e6)
        
        return samples
    
    def sample_votes(self,
                    contestants: pd.DataFrame,
                    n_samples: Optional[int] = None) -> np.ndarray:
        """
        从后验分布采样投票（兼容旧接口）
        """
        return self.sample_votes_posterior(contestants, n_samples)
    
    def estimate_votes(self,
                      contestants: pd.DataFrame,
                      total_votes: float = 1e6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        估计投票及可信区间
        
        Args:
            contestants: 选手数据
            total_votes: 总投票数
        
        Returns:
            (后验均值, 2.5%分位数, 97.5%分位数)
        """
        samples = self.sample_votes_posterior(contestants)
        
        # 重新归一化
        for i in range(len(samples)):
            samples[i] = normalize_votes(samples[i], total_votes)
        
        mean_votes = samples.mean(axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        
        return mean_votes, lower, upper
    
    def _evaluate_accuracy(self, 
                          weekly_data: pd.DataFrame,
                          elimination_info: pd.DataFrame) -> float:
        """评估淘汰预测准确率"""
        correct = 0
        total = 0
        
        for (season, week), group in weekly_data.groupby(['season', 'week']):
            if len(group) <= 1:
                continue
            
            elim = elimination_info[
                (elimination_info['season'] == season) &
                (elimination_info['week'] == week)
            ]
            
            if len(elim) == 0:
                continue
            
            actual_name = elim.iloc[0]['eliminated_name']
            
            # 预测
            mean_votes, _, _ = self.estimate_votes(group)
            scores = group['total_score'].values
            
            # 使用自动选择的方法
            if season in [1, 2] or season >= 28:
                # 排名法
                from src.utils import compute_rank_combined_score, get_eliminated_index_rank
                combined = compute_rank_combined_score(scores, mean_votes)
                pred_idx = get_eliminated_index_rank(combined)
            else:
                # 百分比法
                from src.utils import compute_percent_combined_score, get_eliminated_index_percent
                combined = compute_percent_combined_score(scores, mean_votes)
                pred_idx = get_eliminated_index_percent(combined)
            
            pred_name = group.iloc[pred_idx]['celebrity_name']
            
            if pred_name == actual_name:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def estimate_all_weeks(self,
                          season_week_data: Dict,
                          total_votes: float = 1e6) -> Dict:
        """
        估计所有周次的投票
        
        Args:
            season_week_data: 赛季-周次数据
            total_votes: 每周总投票数
        
        Returns:
            估计结果字典
        """
        self.estimates = {}
        self.samples = {}
        
        for (season, week), contestants in season_week_data.items():
            if len(contestants) <= 1:
                continue
            
            samples = self.sample_votes(contestants)
            mean_votes, lower, upper = self.estimate_votes(contestants, total_votes)
            
            self.estimates[(season, week)] = {
                'names': contestants['celebrity_name'].values,
                'scores': contestants['total_score'].values,
                'votes': mean_votes,
                'lower': lower,
                'upper': upper
            }
            self.samples[(season, week)] = samples
        
        return self.estimates
    
    def get_uncertainty_stats(self, season: int, week: int) -> Dict:
        """
        获取某周的不确定性统计
        
        Args:
            season: 赛季
            week: 周次
        
        Returns:
            不确定性统计字典
        """
        key = (season, week)
        if key not in self.estimates:
            return {}
        
        est = self.estimates[key]
        samples = self.samples.get(key, None)
        
        result = {
            'names': est['names'],
            'mean': est['votes'],
            'lower': est['lower'],
            'upper': est['upper'],
            'ci_width': est['upper'] - est['lower']
        }
        
        if samples is not None:
            result['std'] = samples.std(axis=0)
            result['cv'] = result['std'] / result['mean']
        
        return result
