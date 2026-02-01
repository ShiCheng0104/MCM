"""
模型评估与推广 - Q3模型评估
Factor Impact Analysis Model Evaluation
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_evaluation.config import Q3_PATH, EVALUATION_CONFIG, DATA_PATH


class Q3ModelEvaluator:
    """Q3: 影响因素分析模型评估"""
    
    def __init__(self):
        self.load_data()
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        # 加载原始数据
        self.raw_data = pd.read_csv(DATA_PATH)
        
        # 加载效应对比结果
        effect_path = os.path.join(Q3_PATH, 'outputs', 'tables', 'effect_comparison.csv')
        self.effect_comparison = pd.read_csv(effect_path)
        
        # 加载特征重要性
        score_importance_path = os.path.join(Q3_PATH, 'outputs', 'tables', 
                                              'judge_score_feature_importance.csv')
        vote_importance_path = os.path.join(Q3_PATH, 'outputs', 'tables', 
                                             'fan_vote_feature_importance.csv')
        
        if os.path.exists(score_importance_path):
            self.score_importance = pd.read_csv(score_importance_path)
        if os.path.exists(vote_importance_path):
            self.vote_importance = pd.read_csv(vote_importance_path)
            
        # 加载各效应数据
        self.partner_effect = pd.read_csv(os.path.join(Q3_PATH, 'outputs', 'tables', 'partner_effect.csv'))
        self.age_effect = pd.read_csv(os.path.join(Q3_PATH, 'outputs', 'tables', 'age_effect.csv'))
        self.industry_effect = pd.read_csv(os.path.join(Q3_PATH, 'outputs', 'tables', 'industry_effect.csv'))
        
    def prepare_features(self):
        """准备特征矩阵"""
        df = self.raw_data.copy()
        
        # 过滤有效数据
        df = df[df['total_score'].notna()].copy()
        
        # 创建特征
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100],
                                 labels=['Young', 'Prime', 'Middle', 'Mature', 'Senior'])
        
        # 编码分类变量
        features = pd.get_dummies(df[['age', 'season', 'week']], drop_first=True)
        
        # 目标变量
        y_score = df['total_score']
        
        return features, y_score, df
    
    def cross_validation_score(self):
        """交叉验证评分"""
        X, y_score, _ = self.prepare_features()
        
        n_folds = EVALUATION_CONFIG['cv_folds']
        
        # 随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # 交叉验证
        cv_r2 = cross_val_score(rf_model, X, y_score, cv=n_folds, scoring='r2')
        cv_rmse = -cross_val_score(rf_model, X, y_score, cv=n_folds, 
                                   scoring='neg_root_mean_squared_error')
        
        self.results['cross_validation'] = {
            'r2_mean': cv_r2.mean(),
            'r2_std': cv_r2.std(),
            'r2_scores': cv_r2.tolist(),
            'rmse_mean': cv_rmse.mean(),
            'rmse_std': cv_rmse.std(),
            'n_folds': n_folds,
        }
        
        return self.results['cross_validation']
    
    def feature_importance_stability(self):
        """特征重要性稳定性分析"""
        X, y_score, _ = self.prepare_features()
        
        n_bootstrap = 50  # 减少迭代次数以加快速度
        importance_matrix = []
        
        for i in range(n_bootstrap):
            # Bootstrap采样
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X.iloc[idx]
            y_boot = y_score.iloc[idx]
            
            # 训练模型
            rf = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
            rf.fit(X_boot, y_boot)
            
            importance_matrix.append(rf.feature_importances_)
        
        importance_matrix = np.array(importance_matrix)
        
        # 计算稳定性指标
        mean_importance = importance_matrix.mean(axis=0)
        std_importance = importance_matrix.std(axis=0)
        cv_importance = std_importance / (mean_importance + 1e-10)
        
        # 稳定性分数（变异系数的反向）
        stability_score = 1 - np.mean(cv_importance)
        
        self.results['feature_stability'] = {
            'stability_score': stability_score,
            'mean_cv': np.mean(cv_importance),
            'top_features_stable': (cv_importance[:5] < 0.3).mean(),  # 前5个特征是否稳定
        }
        
        return self.results['feature_stability']
    
    def effect_consistency_check(self):
        """效应一致性检验"""
        # 检查各效应是否符合直觉
        
        # 年龄效应检验：预期Prime年龄段表现最好
        age_effects = self.age_effect.copy()
        if 'score_effect' in age_effects.columns:
            score_col = 'score_effect'
        else:
            # 找到包含effect的列
            effect_cols = [c for c in age_effects.columns if 'effect' in c.lower()]
            score_col = effect_cols[0] if effect_cols else age_effects.columns[1]
        
        # 舞伴效应检验：有经验的舞伴应该有正效应
        partner_effects = self.partner_effect.copy()
        
        # 计算效应的一致性指标
        consistency_checks = {
            'age_effect_range': age_effects[score_col].max() - age_effects[score_col].min(),
            'n_positive_age_effects': (age_effects[score_col] > 0).sum(),
            'n_negative_age_effects': (age_effects[score_col] < 0).sum(),
        }
        
        # 效应显著性
        if len(age_effects) >= 2:
            t_stat, p_value = stats.ttest_1samp(age_effects[score_col].dropna(), 0)
            consistency_checks['age_effect_significant'] = p_value < 0.05
            consistency_checks['age_effect_p_value'] = p_value
        
        self.results['effect_consistency'] = consistency_checks
        return self.results['effect_consistency']
    
    def judge_vs_vote_divergence(self):
        """评委得分与观众投票的差异分析"""
        effects = self.effect_comparison.copy()
        
        # 计算score_effect和vote_effect的差异
        if 'effect_difference' in effects.columns:
            divergence = effects['effect_difference'].abs()
        else:
            divergence = (effects['score_effect'] - effects['vote_effect']).abs()
        
        # 相关性分析
        if 'score_effect' in effects.columns and 'vote_effect' in effects.columns:
            correlation = effects['score_effect'].corr(effects['vote_effect'])
        else:
            correlation = 0
        
        self.results['judge_vote_divergence'] = {
            'mean_divergence': divergence.mean(),
            'max_divergence': divergence.max(),
            'correlation': correlation,
            'divergent_factors': effects[divergence > divergence.median()]['category'].tolist() 
                                if 'category' in effects.columns else [],
        }
        
        return self.results['judge_vote_divergence']
    
    def model_comparison(self):
        """模型比较：RF vs Linear Regression"""
        X, y_score, _ = self.prepare_features()
        
        n_folds = EVALUATION_CONFIG['cv_folds']
        
        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X, y_score, cv=n_folds, scoring='r2')
        
        # 线性回归
        lr = LinearRegression()
        lr_scores = cross_val_score(lr, X, y_score, cv=n_folds, scoring='r2')
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(rf_scores, lr_scores)
        
        self.results['model_comparison'] = {
            'random_forest': {
                'mean_r2': rf_scores.mean(),
                'std_r2': rf_scores.std(),
            },
            'linear_regression': {
                'mean_r2': lr_scores.mean(),
                'std_r2': lr_scores.std(),
            },
            'rf_better': rf_scores.mean() > lr_scores.mean(),
            'significant_difference': p_value < 0.05,
            'p_value': p_value,
        }
        
        return self.results['model_comparison']
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("Q3: Factor Impact Analysis Model Evaluation")
        print("=" * 60)
        
        print("\n1. Cross Validation...")
        cv = self.cross_validation_score()
        print(f"   R² Mean: {cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}")
        print(f"   RMSE Mean: {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}")
        
        print("\n2. Feature Importance Stability...")
        fs = self.feature_importance_stability()
        print(f"   Stability Score: {fs['stability_score']:.3f}")
        
        print("\n3. Effect Consistency...")
        ec = self.effect_consistency_check()
        print(f"   Age Effect Range: {ec['age_effect_range']:.3f}")
        
        print("\n4. Judge vs Vote Divergence...")
        jvd = self.judge_vote_divergence()
        print(f"   Correlation: {jvd['correlation']:.3f}")
        print(f"   Mean Divergence: {jvd['mean_divergence']:.3f}")
        
        print("\n5. Model Comparison...")
        mc = self.model_comparison()
        print(f"   RF R²: {mc['random_forest']['mean_r2']:.3f}")
        print(f"   LR R²: {mc['linear_regression']['mean_r2']:.3f}")
        print(f"   RF Better: {mc['rf_better']}, p={mc['p_value']:.4f}")
        
        return self.results
    
    def get_summary_metrics(self):
        """获取摘要指标"""
        if not self.results:
            self.run_full_evaluation()
        
        return {
            'model': 'Q3: Factor Analysis',
            'cv_r2': self.results['cross_validation']['r2_mean'],
            'cv_r2_std': self.results['cross_validation']['r2_std'],
            'rmse': self.results['cross_validation']['rmse_mean'],
            'feature_stability': self.results['feature_stability']['stability_score'],
            'judge_vote_correlation': self.results['judge_vote_divergence']['correlation'],
            'rf_vs_lr_significant': self.results['model_comparison']['significant_difference'],
        }


if __name__ == '__main__':
    evaluator = Q3ModelEvaluator()
    results = evaluator.run_full_evaluation()
    print("\n" + "=" * 60)
    print("Summary:")
    print(evaluator.get_summary_metrics())
