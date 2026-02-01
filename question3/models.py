"""
影响因素分析模型模块
包含多元回归、随机森林和混合效应模型
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import warnings
warnings.filterwarnings('ignore')

from config import RF_PARAMS, RANDOM_SEED


class FeatureEncoder:
    """特征编码器"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def fit_transform(self, df, categorical_cols, numerical_cols):
        """编码特征"""
        encoded_data = {}
        
        # 编码分类变量
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                encoded_data[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # 标准化数值变量
        numerical_data = df[numerical_cols].fillna(0).values
        scaled_data = self.scaler.fit_transform(numerical_data)
        
        for i, col in enumerate(numerical_cols):
            encoded_data[f'{col}_scaled'] = scaled_data[:, i]
        
        self.feature_names = list(encoded_data.keys())
        return pd.DataFrame(encoded_data)
    
    def transform(self, df, categorical_cols, numerical_cols):
        """转换新数据"""
        encoded_data = {}
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # 处理未见过的类别
                values = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                encoded_data[f'{col}_encoded'] = le.transform(values)
        
        numerical_data = df[numerical_cols].fillna(0).values
        scaled_data = self.scaler.transform(numerical_data)
        
        for i, col in enumerate(numerical_cols):
            encoded_data[f'{col}_scaled'] = scaled_data[:, i]
        
        return pd.DataFrame(encoded_data)


class RegressionAnalyzer:
    """多元回归分析器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def fit_ols(self, df, formula, target_name='judge_score'):
        """拟合OLS回归模型"""
        model = ols(formula, data=df).fit()
        self.models[target_name] = model
        self.results[target_name] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'coefficients': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
            'std_errors': model.bse.to_dict(),
            'summary': model.summary()
        }
        return model
    
    def fit_mixed_effects(self, df, formula, groups, target_name='judge_score_mixed'):
        """拟合混合效应模型"""
        try:
            model = mixedlm(formula, df, groups=df[groups]).fit()
            self.models[target_name] = model
            self.results[target_name] = {
                'coefficients': model.fe_params.to_dict(),
                'pvalues': model.pvalues.to_dict(),
                'random_effects_var': model.cov_re.iloc[0, 0] if hasattr(model.cov_re, 'iloc') else None,
                'summary': model.summary()
            }
            return model
        except Exception as e:
            print(f"混合效应模型拟合失败: {e}")
            return None
    
    def get_coefficient_comparison(self):
        """获取系数对比表"""
        comparison = {}
        for name, result in self.results.items():
            comparison[name] = {
                'coefficients': result['coefficients'],
                'pvalues': result['pvalues']
            }
        return comparison


class RandomForestAnalyzer:
    """随机森林分析器"""
    
    def __init__(self, params=None):
        self.params = params or RF_PARAMS
        self.models = {}
        self.feature_importance = {}
        self.encoder = FeatureEncoder()
    
    def prepare_features(self, df, categorical_cols, numerical_cols):
        """准备特征矩阵"""
        # One-hot编码分类变量
        X_cat = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, drop_first=True)
        
        # 数值变量
        X_num = df[numerical_cols].fillna(0)
        
        # 合并
        X = pd.concat([X_cat, X_num], axis=1)
        
        return X
    
    def fit(self, X, y, target_name='target'):
        """拟合随机森林模型"""
        model = RandomForestRegressor(**self.params)
        model.fit(X, y)
        
        self.models[target_name] = model
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[target_name] = importance
        
        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        return {
            'model': model,
            'feature_importance': importance,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'train_r2': r2_score(y, model.predict(X))
        }
    
    def compare_importance(self):
        """对比不同目标的特征重要性"""
        comparison = {}
        for name, imp in self.feature_importance.items():
            comparison[name] = imp.set_index('feature')['importance'].to_dict()
        return pd.DataFrame(comparison)


class EffectAnalyzer:
    """效应分析器 - 量化各因素影响"""
    
    def __init__(self, df):
        self.df = df
        self.effects = {}
    
    def calculate_partner_effect(self):
        """计算舞伴效应"""
        # 计算每个舞伴的选手平均得分与整体平均的差异
        overall_mean_score = self.df['total_score'].mean()
        overall_mean_votes = self.df['estimated_votes'].mean()
        
        partner_effects = []
        for partner in self.df['partner'].unique():
            partner_data = self.df[self.df['partner'] == partner]
            
            if len(partner_data) >= 3:  # 至少有3条记录
                effect = {
                    'partner': partner,
                    'num_celebrities': partner_data['celebrity'].nunique(),
                    'num_seasons': partner_data['season'].nunique(),
                    'avg_score': partner_data['total_score'].mean(),
                    'avg_votes': partner_data['estimated_votes'].mean(),
                    'avg_placement': partner_data['placement'].mean(),
                    'score_effect': partner_data['total_score'].mean() - overall_mean_score,
                    'vote_effect': partner_data['estimated_votes'].mean() - overall_mean_votes,
                    'score_effect_pct': (partner_data['total_score'].mean() - overall_mean_score) / overall_mean_score * 100,
                    'vote_effect_pct': (partner_data['estimated_votes'].mean() - overall_mean_votes) / overall_mean_votes * 100,
                    'win_count': (partner_data.groupby('celebrity')['placement'].first() == 1).sum(),
                    'top3_count': (partner_data.groupby('celebrity')['placement'].first() <= 3).sum()
                }
                partner_effects.append(effect)
        
        self.effects['partner'] = pd.DataFrame(partner_effects).sort_values('score_effect', ascending=False)
        return self.effects['partner']
    
    def calculate_age_effect(self):
        """计算年龄效应"""
        age_effects = self.df.groupby('age_group').agg({
            'total_score': ['mean', 'std'],
            'estimated_votes': ['mean', 'std'],
            'placement': 'mean',
            'celebrity': 'nunique'
        }).reset_index()
        
        age_effects.columns = [
            'age_group', 'avg_score', 'score_std',
            'avg_votes', 'vote_std', 'avg_placement', 'num_celebrities'
        ]
        
        # 计算相对效应
        overall_mean_score = self.df['total_score'].mean()
        overall_mean_votes = self.df['estimated_votes'].mean()
        
        age_effects['score_effect'] = age_effects['avg_score'] - overall_mean_score
        age_effects['vote_effect'] = age_effects['avg_votes'] - overall_mean_votes
        
        self.effects['age'] = age_effects
        return self.effects['age']
    
    def calculate_industry_effect(self):
        """计算行业效应"""
        industry_effects = self.df.groupby('industry_simplified').agg({
            'total_score': ['mean', 'std'],
            'estimated_votes': ['mean', 'std'],
            'placement': 'mean',
            'celebrity': 'nunique'
        }).reset_index()
        
        industry_effects.columns = [
            'industry', 'avg_score', 'score_std',
            'avg_votes', 'vote_std', 'avg_placement', 'num_celebrities'
        ]
        
        # 计算相对效应
        overall_mean_score = self.df['total_score'].mean()
        overall_mean_votes = self.df['estimated_votes'].mean()
        
        industry_effects['score_effect'] = industry_effects['avg_score'] - overall_mean_score
        industry_effects['vote_effect'] = industry_effects['avg_votes'] - overall_mean_votes
        
        self.effects['industry'] = industry_effects.sort_values('avg_placement')
        return self.effects['industry']
    
    def calculate_domestic_effect(self):
        """计算国内/国际选手效应"""
        domestic_effects = self.df.groupby('is_domestic').agg({
            'total_score': ['mean', 'std'],
            'estimated_votes': ['mean', 'std'],
            'placement': 'mean',
            'celebrity': 'nunique'
        }).reset_index()
        
        domestic_effects.columns = [
            'is_domestic', 'avg_score', 'score_std',
            'avg_votes', 'vote_std', 'avg_placement', 'num_celebrities'
        ]
        
        domestic_effects['is_domestic'] = domestic_effects['is_domestic'].map({0: 'International', 1: 'Domestic (US)'})
        
        self.effects['domestic'] = domestic_effects
        return self.effects['domestic']
    
    def calculate_all_effects(self):
        """计算所有效应"""
        self.calculate_partner_effect()
        self.calculate_age_effect()
        self.calculate_industry_effect()
        self.calculate_domestic_effect()
        return self.effects
    
    def get_effect_comparison(self):
        """获取评委得分与观众投票效应对比"""
        comparison = {
            'factor': [],
            'category': [],
            'score_effect': [],
            'vote_effect': [],
            'effect_difference': []
        }
        
        # 年龄效应
        if 'age' in self.effects:
            age_df = self.effects['age']
            for _, row in age_df.iterrows():
                comparison['factor'].append('Age')
                comparison['category'].append(str(row['age_group']))
                score_eff = row.get('score_effect', row['avg_score'] - self.df['total_score'].mean())
                vote_eff = row.get('vote_effect', row['avg_votes'] - self.df['estimated_votes'].mean())
                comparison['score_effect'].append(score_eff)
                comparison['vote_effect'].append(vote_eff)
                comparison['effect_difference'].append(vote_eff - score_eff)
        
        # 行业效应
        if 'industry' in self.effects:
            industry_df = self.effects['industry']
            for _, row in industry_df.iterrows():
                comparison['factor'].append('Industry')
                comparison['category'].append(str(row['industry']))
                score_eff = row.get('score_effect', row['avg_score'] - self.df['total_score'].mean())
                vote_eff = row.get('vote_effect', row['avg_votes'] - self.df['estimated_votes'].mean())
                comparison['score_effect'].append(score_eff)
                comparison['vote_effect'].append(vote_eff)
                comparison['effect_difference'].append(vote_eff - score_eff)
        
        # 国籍效应
        if 'domestic' in self.effects:
            domestic_df = self.effects['domestic']
            overall_mean_score = self.df['total_score'].mean()
            overall_mean_votes = self.df['estimated_votes'].mean()
            for _, row in domestic_df.iterrows():
                comparison['factor'].append('Nationality')
                comparison['category'].append(str(row['is_domestic']))
                score_eff = row['avg_score'] - overall_mean_score
                vote_eff = row['avg_votes'] - overall_mean_votes
                comparison['score_effect'].append(score_eff)
                comparison['vote_effect'].append(vote_eff)
                comparison['effect_difference'].append(vote_eff - score_eff)
        
        return pd.DataFrame(comparison)


if __name__ == '__main__':
    from data_loader import create_analysis_dataset
    
    # 测试
    df = create_analysis_dataset()
    
    # 效应分析
    analyzer = EffectAnalyzer(df)
    effects = analyzer.calculate_all_effects()
    
    print("舞伴效应 (Top 10):")
    print(effects['partner'].head(10))
    
    print("\n年龄效应:")
    print(effects['age'])
    
    print("\n行业效应:")
    print(effects['industry'])
