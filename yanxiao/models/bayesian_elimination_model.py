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
                 n_tune: int = 3000,
                 n_chains: int = 4,
                 target_accept: float = 0.99,
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
        self.beta_pop = 0.5   # 知名度效应
        self.beta_viewers = 1.0   # 观众人数效应                self.beta_momentum = 0.0  # 动量效应（本周-上周）
        self.beta_trend = 0.0     # 趋势效应（最近3周平均）
        self.beta_history = 0.0   # 历史表现（累积平均）        self.sigma = 0.5          # 噪声标准差
        
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
        self.pop_mean = 0
        self.pop_std = 1
        self.viewers_mean = 0
        self.viewers_std = 1
        self.momentum_mean = 0
        self.momentum_std = 1
        self.trend_mean = 0
        self.trend_std = 1
        self.history_mean = 0
        self.history_std = 1
        
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
        
        # 加载观众人数数据
        self._load_viewership_data(weekly_data)
        
        # 添加时间序列特征
        self._add_time_series_features(weekly_data)
        
        # 尝试加载Google Trends数据
        try:
            # 从新的路径加载数据
            trends_path = 'MCM/Googletrends/google_trends_summary.csv'
            if not pd.io.common.file_exists(trends_path):
                trends_path = 'Googletrends/google_trends_summary.csv'
            
            trends_df = pd.read_csv(trends_path)
            
            # 确定join列
            if 'celebrity_name' in weekly_data.columns:
                join_col = 'celebrity_name'
            else:
                join_col = 'celebrity'
            
            # 使用normalized_average列（这是按季度归一化的数据）
            if 'normalized_average' in trends_df.columns:
                # 建立映射：需要同时匹配celebrity_name和season
                # 因为Google Trends数据是按季度归一化的
                weekly_data['popularity'] = 0.0
                
                for idx, row in weekly_data.iterrows():
                    celeb = row[join_col]
                    season = row['season']
                    
                    # 查找对应的trends数据
                    match = trends_df[
                        (trends_df['celebrity_name'] == celeb) & 
                        (trends_df['season'] == season)
                    ]
                    
                    if len(match) > 0:
                        weekly_data.at[idx, 'popularity'] = match.iloc[0]['normalized_average']
                
                # 填充缺失值（用季度内的中位数）
                for season in weekly_data['season'].unique():
                    season_mask = weekly_data['season'] == season
                    season_median = weekly_data.loc[season_mask, 'popularity'].median()
                    if pd.notna(season_median) and season_median > 0:
                        weekly_data.loc[season_mask & (weekly_data['popularity'] == 0), 'popularity'] = season_median
                
                # Google Trends数据已经是0-1归一化的，直接使用即可
                # 注意：这里不再进行log变换，因为数据已经是归一化的比例
                
                print(f"  已合并Google Trends知名度数据（按季度归一化，范围0-1）")
                
        except Exception as e:
            print(f"  无法加载Google Trends数据 (使用默认值0): {e}")
            weekly_data['popularity'] = 0
        
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
        
        # 标准化知名度
        if 'popularity' in weekly_data.columns:
            self.pop_mean = weekly_data['popularity'].mean()
            self.pop_std = weekly_data['popularity'].std()
            if self.pop_std == 0: self.pop_std = 1
            # 注意：Google Trends数据已经是0-1归一化的，这里的标准化只是为了统一尺度
        
        # 标准化观众人数
        if 'viewers' in weekly_data.columns:
            self.viewers_mean = weekly_data['viewers'].mean()
            self.viewers_std = weekly_data['viewers'].std()
            if self.viewers_std == 0: self.viewers_std = 1
        
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
                
                # 知名度
                pop_val = row.get('popularity', 0)
                pop_norm = (pop_val - self.pop_mean) / self.pop_std
                
                # 观众人数
                viewers_val = row.get('viewers', 0)
                viewers_norm = (viewers_val - self.viewers_mean) / self.viewers_std
                
                # 时间序列特征
                momentum_val = row.get('momentum', 0)
                momentum_norm = (momentum_val - self.momentum_mean) / self.momentum_std
                
                trend_val = row.get('trend', 0)
                trend_norm = (trend_val - self.trend_mean) / self.trend_std
                
                history_val = row.get('history', row['total_score'])
                history_norm = (history_val - self.history_mean) / self.history_std
                
                contestants_data.append({
                    'name': row[name_col],
                    'score': row['total_score'],
                    'score_norm': (row['total_score'] - self.score_mean) / self.score_std,
                    'age': age_val if pd.notna(age_val) else 30,
                    'age_norm': (age_val - self.age_mean) / self.age_std if (age_col and pd.notna(age_val)) else 0,
                    'pop_norm': pop_norm,
                    'viewers_norm': viewers_norm,
                    'momentum_norm': momentum_norm,
                    'trend_norm': trend_norm,
                    'history_norm': history_norm,
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
                    'pop_norm': c.get('pop_norm', 0),
                    'viewers_norm': c.get('viewers_norm', 0),
                    'momentum_norm': c.get('momentum_norm', 0),
                    'trend_norm': c.get('trend_norm', 0),
                    'history_norm': c.get('history_norm', 0),
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
            
            # --- 科学调优：分层先验结构 (Hierarchical Priors) ---
            
            # 1. 超参数：不再手动指定固定sigma，而是从数据中学习参数的波动范围
            # 使用HalfPreNormal作为方差的先验，这是一种"弱信息先验"，允许数据主导
            sigma_beta_score = pm.HalfNormal('sigma_beta_score', sigma=1.0)
            sigma_beta_pop = pm.HalfNormal('sigma_beta_pop', sigma=1.0)
            
            # 2. 固定效应：放宽均值限制，引入分层方差
            # beta_0: 保持原状，截距项通常比较稳定
            beta_0 = pm.Normal('beta_0', mu=10, sigma=5) 
            
            # beta_score: 均值0.5 (正向)，但允许更大的不确定性，且方差由数据决定
            beta_score = pm.TruncatedNormal('beta_score', mu=0.5, sigma=sigma_beta_score, lower=-0.5)
            
            # beta_age: 允许微弱的负向影响
            beta_age = pm.Normal('beta_age', mu=0, sigma=0.5)
            
            # beta_pop: 知名度效应。均值设为1.0作为初始猜测，但允许更宽的波动
            # 同时将lower设为-0.5，允许少量的"黑红"效应(高关注度也可能是负面)
            beta_pop = pm.Normal('beta_pop', mu=0.5, sigma=sigma_beta_pop)            
            # beta_viewers: 观众人数效应。观众人数越多，可能投票总数越高
            # 使用分层先验，允许数据主导
            sigma_beta_viewers = pm.HalfNormal('sigma_beta_viewers', sigma=1.0)
            beta_viewers = pm.Normal('beta_viewers', mu=0.5, sigma=sigma_beta_viewers)
            
            # 时间序列效应
            # beta_momentum: 动量效应（本周比上周进步→更多票）
            beta_momentum = pm.Normal('beta_momentum', mu=0.3, sigma=0.5)
            
            # beta_trend: 趋势效应（持续进步趋势→更多票）
            beta_trend = pm.Normal('beta_trend', mu=0.2, sigma=0.5)
            
            # beta_history: 历史表现效应（过往表现好→更多票）
            beta_history = pm.Normal('beta_history', mu=0.3, sigma=0.5)            
            # [修正] 移除可能导致数值错误的二次log变换
            # 数据预处理中已经进行了log1p变换，这里使用线性效应即可
            # 这种分层先验结构本身就已经足够灵活

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
            mu = (beta_0 + 
                  beta_score * df['score_norm'].values + 
                  beta_age * df['age_norm'].values + 
                  beta_pop * df['pop_norm'].values +
                  beta_viewers * df['viewers_norm'].values +
                  beta_momentum * df['momentum_norm'].values +
                  beta_trend * df['trend_norm'].values +
                  beta_history * df['history_norm'].values)
            
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
        summary = az.summary(self.trace, var_names=['beta_0', 'beta_score', 'beta_age', 'beta_pop', 'beta_viewers', 
                                                     'beta_momentum', 'beta_trend', 'beta_history', 'sigma'])
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
            beta_pop = params[3]
            beta_viewers = params[4]
            beta_momentum = params[5]
            beta_trend = params[6]
            beta_history = params[7]
            sigma = max(params[8], 0.1)
            
            idx = 9
            partner_eff = params[idx:idx+n_partners] if n_partners > 0 else []
            season_eff = params[idx+n_partners:idx+n_partners+n_seasons] if n_seasons > 0 else []
            industry_eff = params[idx+n_partners+n_seasons:] if n_industries > 0 else []
            
            nll = 0.0
            
            for week_data in train_data:
                contestants = week_data['contestants']
                eliminated_names = week_data['eliminated_names']
                
                if len(contestants) < 2:
                    continue
                
                # 计算log投票
                log_votes = []
                for c in contestants:
                    lv = (beta_0 + 
                          beta_score * c['score_norm'] + 
                          beta_age * c['age_norm'] + 
                          beta_pop * c.get('pop_norm', 0) +
                          beta_viewers * c.get('viewers_norm', 0) +
                          beta_momentum * c.get('momentum_norm', 0) +
                          beta_trend * c.get('trend_norm', 0) +
                          beta_history * c.get('history_norm', 0))
                    
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
            nll += reg * (beta_score**2 + beta_age**2 + beta_pop**2 + beta_viewers**2 +
                         beta_momentum**2 + beta_trend**2 + beta_history**2)
            if n_partners > 0:
                nll += reg * np.sum(np.array(partner_eff)**2)
            if n_seasons > 0:
                nll += reg * np.sum(np.array(season_eff)**2)
            if n_industries > 0:
                nll += reg * np.sum(np.array(industry_eff)**2)
            
            return nll
        
        # 初始化
        n_params = 9 + n_partners + n_seasons + n_industries
        x0 = np.zeros(n_params)
        x0[0] = 10.0  # beta_0
        x0[1] = 0.5   # beta_score
        x0[3] = 1.0   # beta_pop
        x0[4] = 0.5   # beta_viewers
        x0[5] = 0.3   # beta_momentum
        x0[6] = 0.2   # beta_trend
        x0[7] = 0.3   # beta_history
        x0[8] = 0.5   # sigma
        
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
        self.beta_pop = params[3]
        self.beta_viewers = params[4]
        self.beta_momentum = params[5]
        self.beta_trend = params[6]
        self.beta_history = params[7]
        self.sigma = max(params[8], 0.1)
        
        idx = 9
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
        print(f"    β_pop = {self.beta_pop:.3f}")
        print(f"    β_viewers = {self.beta_viewers:.3f}")
        print(f"    β_momentum = {self.beta_momentum:.3f}")
        print(f"    β_trend = {self.beta_trend:.3f}")
        print(f"    β_history = {self.beta_history:.3f}")
        print(f"    σ = {self.sigma:.3f}")
    
    def _extract_posterior_stats(self):
        """从PyMC trace提取后验统计量"""
        posterior = self.trace.posterior
        
        self.beta_0 = float(posterior['beta_0'].mean())
        self.beta_score = float(posterior['beta_score'].mean())
        self.beta_age = float(posterior['beta_age'].mean())
        self.beta_pop = float(posterior['beta_pop'].mean())
        self.beta_viewers = float(posterior['beta_viewers'].mean())
        self.beta_momentum = float(posterior['beta_momentum'].mean())
        self.beta_trend = float(posterior['beta_trend'].mean())
        self.beta_history = float(posterior['beta_history'].mean())
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
        print(f"    β_pop = {self.beta_pop:.3f} (知名度效应系数)")
        print(f"    β_viewers = {self.beta_viewers:.3f} (观众人数效应系数)")
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
                
                pop_val = row.get('popularity', 0)
                pop_norm = (pop_val - self.pop_mean) / self.pop_std
                
                viewers_val = row.get('viewers', 0)
                viewers_norm = (viewers_val - self.viewers_mean) / self.viewers_std
                
                momentum_val = row.get('momentum', 0)
                momentum_norm = (momentum_val - self.momentum_mean) / self.momentum_std
                
                trend_val = row.get('trend', 0)
                trend_norm = (trend_val - self.trend_mean) / self.trend_std
                
                history_val = row.get('history', score)
                history_norm = (history_val - self.history_mean) / self.history_std
                
                # 计算log投票
                log_vote = (self.beta_0 + 
                           self.beta_score * score_norm + 
                           self.beta_age * age_norm + 
                           self.beta_pop * pop_norm +
                           self.beta_viewers * viewers_norm +
                           self.beta_momentum * momentum_norm +
                           self.beta_trend * trend_norm +
                           self.beta_history * history_norm)
                
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
        
        # 分析不确定性
        if len(self.results_df) > 0:
            cv = self.results_df['vote_std'] / self.results_df['estimated_votes']
            print("\n  投票估计确定性分析 (变异系数 CV = std/mean):")
            print(f"    平均 CV: {cv.mean():.4f}")
            print(f"    高确定性 (CV < 0.3) 占比: {(cv < 0.3).mean():.2%}")
            print(f"    中确定性 (CV < 0.5) 占比: {(cv < 0.5).mean():.2%}")
            
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
    
    def _add_time_series_features(self, weekly_data: pd.DataFrame):
        """添加时间序列特征"""
        print("  正在计算时间序列特征...")
        
        # 确定列名
        name_col = 'celebrity_name' if 'celebrity_name' in weekly_data.columns else 'celebrity'
        
        # 初始化特征列
        weekly_data['momentum'] = 0.0  # 分数动量
        weekly_data['trend'] = 0.0     # 分数趋势
        weekly_data['history'] = 0.0   # 历史平均
        
        # 按选手和赛季分组计算
        for (season, celeb), group in weekly_data.groupby(['season', name_col]):
            if len(group) < 2:
                continue
            
            # 按周排序
            group = group.sort_values('week')
            indices = group.index
            scores = group['total_score'].values
            
            for i, idx in enumerate(indices):
                week = group.loc[idx, 'week']
                
                # 1. 动量特征：本周分数 - 上周分数
                if i > 0:
                    momentum = scores[i] - scores[i-1]
                    weekly_data.at[idx, 'momentum'] = momentum
                
                # 2. 趋势特征：最近3周的线性趋势（斜率）
                if i >= 2:
                    recent_scores = scores[max(0, i-2):i+1]
                    recent_weeks = list(range(len(recent_scores)))
                    # 简单线性拟合斜率
                    if len(recent_scores) >= 2:
                        trend = np.polyfit(recent_weeks, recent_scores, 1)[0]
                        weekly_data.at[idx, 'trend'] = trend
                
                # 3. 历史表现：截至上周的累积平均（不包括本周）
                if i > 0:
                    history_avg = np.mean(scores[:i])
                    weekly_data.at[idx, 'history'] = history_avg
                else:
                    # 第一周使用本周分数作为历史
                    weekly_data.at[idx, 'history'] = scores[i]
        
        # 标准化时间序列特征
        self.momentum_mean = weekly_data['momentum'].mean()
        self.momentum_std = weekly_data['momentum'].std()
        if self.momentum_std == 0: self.momentum_std = 1
        
        self.trend_mean = weekly_data['trend'].mean()
        self.trend_std = weekly_data['trend'].std()
        if self.trend_std == 0: self.trend_std = 1
        
        self.history_mean = weekly_data['history'].mean()
        self.history_std = weekly_data['history'].std()
        if self.history_std == 0: self.history_std = 1
        
        print(f"    动量特征: mean={self.momentum_mean:.3f}, std={self.momentum_std:.3f}")
        print(f"    趋势特征: mean={self.trend_mean:.3f}, std={self.trend_std:.3f}")
        print(f"    历史特征: mean={self.history_mean:.3f}, std={self.history_std:.3f}")
    
    def _load_viewership_data(self, weekly_data: pd.DataFrame):
        """加载观众人数数据"""
        import os
        
        print("  正在加载观众人数数据...")
        
        # 读取所有季度的观众人数文件
        viewership_dir = 'MCM/收视率数据/processed'
        if not os.path.exists(viewership_dir):
            viewership_dir = '收视率数据/processed'
        
        season_viewership = {}  # {season: {week: viewers}}
        
        for season in range(1, 35):
            # 优先读取merged文件，如果不存在则读取普通文件
            merged_file = os.path.join(viewership_dir, f'processed_ratings_{season}_merged.csv')
            normal_file = os.path.join(viewership_dir, f'processed_ratings_{season}.csv')
            
            file_to_read = merged_file if os.path.exists(merged_file) else normal_file
            
            if os.path.exists(file_to_read):
                try:
                    df = pd.read_csv(file_to_read, usecols=['week', 'viewers'])
                    season_viewership[season] = {}
                    
                    for _, row in df.iterrows():
                        week = row['week']
                        viewers_str = str(row['viewers'])
                        
                        # 处理逗号分割的多次播出数据（merged文件）
                        if ',' in viewers_str:
                            # 取多次播出的平均值
                            viewer_values = [float(v.strip()) for v in viewers_str.split(',')]
                            avg_viewers = np.mean(viewer_values)
                        else:
                            avg_viewers = float(viewers_str)
                        
                        season_viewership[season][week] = avg_viewers
                        
                except Exception as e:
                    print(f"    警告: 读取Season {season}观众数据失败: {e}")
        
        # 补充缺失的第12和31季数据（用相邻季度平均值）
        for missing_season in [12, 31]:
            if missing_season not in season_viewership:
                print(f"    补充Season {missing_season}缺失数据（使用相邻季度平均值）")
                
                # 找到相邻季度
                prev_season = missing_season - 1
                next_season = missing_season + 1
                
                if prev_season in season_viewership and next_season in season_viewership:
                    season_viewership[missing_season] = {}
                    
                    # 获取所有可能的周次
                    all_weeks = set(season_viewership[prev_season].keys()) | set(season_viewership[next_season].keys())
                    
                    for week in all_weeks:
                        prev_val = season_viewership[prev_season].get(week, None)
                        next_val = season_viewership[next_season].get(week, None)
                        
                        if prev_val is not None and next_val is not None:
                            season_viewership[missing_season][week] = (prev_val + next_val) / 2
                        elif prev_val is not None:
                            season_viewership[missing_season][week] = prev_val
                        elif next_val is not None:
                            season_viewership[missing_season][week] = next_val
        
        # 将观众人数映射到weekly_data
        viewers_list = []
        missing_count = 0
        for _, row in weekly_data.iterrows():
            season = row['season']
            week = row['week']
            
            if season in season_viewership and week in season_viewership[season]:
                viewers_list.append(season_viewership[season][week])
            else:
                viewers_list.append(np.nan)
                missing_count += 1
        
        weekly_data['viewers'] = viewers_list
        
        # 填充缺失值（用中位数）
        median_viewers = weekly_data['viewers'].median()
        weekly_data['viewers'] = weekly_data['viewers'].fillna(median_viewers)
        
        # 对观众人数进行对数变换（因为投票数和观众数都是乘性关系）
        weekly_data['viewers'] = np.log1p(weekly_data['viewers'])
        
        print(f"    成功加载观众人数数据，覆盖{len(season_viewership)}个季度")
        print(f"    缺失值数量: {missing_count}/{len(viewers_list)} ({missing_count/len(viewers_list):.2%})")
