
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys

# Configure fonts for Chinese support
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_analysis_dataset
from models import RandomForestAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colors from 配色.md
MCM_COLORS = ['#264653', '#2a9d8e', '#e9c46b', '#f3a261', '#e86f52']

# Create a continuous cmap for SHAP Analysis
# Following XGBOOST+SHAP.R style (Yellow to Red/Purple), using our palette colors
# e9c46b (Yellow) to e86f52 (Red Orange) to 264653 (Dark Teal/Purple-ish)
# Low -> Medium -> High
SHAP_CMAP = LinearSegmentedColormap.from_list("mcm_shap", ['#e9c46b', '#e86f52', '#264653'])

def train_rf_model(X, y):
    """Train a Random Forest model"""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

def plot_shap_analysis(model, X, target_name="Target"):
    """
    Generate SHAP plots for Random Forest
    """
    try:
        import shap
    except ImportError:
        print("未安装 shap 库，正在跳过 SHAP 分析。请运行 `pip install shap` 安装。")
        return

    print(f"Generating SHAP plots for {target_name}...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 10))
    # Using 'layered_violin' or 'dot' (beeswarm) style
    # Adjust plot to match R beeswarm style: clear grid, nice colors
    shap.summary_plot(shap_values, X, show=False, plot_type="dot", cmap=SHAP_CMAP, alpha=0.8)
    
    # Customize title and axis
    plt.title(f"{target_name} - SHAP Summary (Beeswarm)", fontsize=16, color='#333333', pad=20)
    plt.xlabel("SHAP value (Impact on model output)", fontsize=12)
    
    # Add grid manually since shap might override
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'shap_summary_{target_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, color=MCM_COLORS[0])
    plt.title(f"{target_name} - Feature Importance", fontsize=16, color='#333333', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'shap_importance_{target_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP plots saved to {FIGURES_DIR}")

def plot_static_nomogram(X, y, target_name="Target"):
    """
    Generate a static Nomogram-like visualization for a Linear Model
    """
    print(f"Generating Nomogram for {target_name}...")
    
    # 1. Fit Linear Model (OLS)
    # Handle categorical variables manually for better display if needed, 
    # but here we use the prepared X which is already one-hot encoded or numeric.
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index=X.columns)
    intercept = model.intercept_
    
    # 2. Calculate contribution ranges
    feature_ranges = {}
    feature_points = {}
    
    # Calculate impact of each feature: value * coef
    max_impact = 0
    
    # For one-hot encoded features, group them? 
    # For simplicity, we stick to the columns in X.
    
    # Iterate to find min/max contribution of each feature
    contributions = []

    # Ensure X is numeric (float) to avoid boolean subtraction errors
    X = X.astype(float)
    
    for col in X.columns:
        vals = X[col]
        min_val = vals.min()
        max_val = vals.max()
        
        # Contribution range
        eff1 = min_val * coefs[col]
        eff2 = max_val * coefs[col]
        
        raw_range = abs(eff2 - eff1)
        contributions.append({
            'feature': col,
            'min_val': min_val,
            'max_val': max_val,
            'coef': coefs[col],
            'range': raw_range,
            'eff_min': min(eff1, eff2),
            'eff_max': max(eff1, eff2)
        })
    
    contributions_df = pd.DataFrame(contributions)
    
    # Normalize to 0-100 points
    max_range = contributions_df['range'].max()
    if max_range == 0:
        print("Model coefficients are all zero, cannot plot nomogram.")
        return

    # Scale factor: 100 points = max_range
    scale = 100.0 / max_range
    
    # Prepare Data for Plotting
    # We will plot "Scales" for top N features
    top_features = contributions_df.sort_values('range', ascending=False).head(10) # Top 10 features
    
    # Increase figure height to accommodate curves
    # Adjust figure size for compactness (reduced height multiplier)
    row_height = 0.8  # Reduced from 1.5
    fig, ax = plt.subplots(figsize=(14, len(top_features) * row_height + 4))
    
    # Background Color (Light Warm/White)
    bg_color = '#F8F9FA' # Light grey background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Colors from 配色.md
    # 1: #264653, 2: #2a9d8e, 3: #e9c46b, 4: #f3a261, 5: #e86f52
    color_lines = '#264653' # Dark Blue/Cyan for structure
    color_text = '#264653'
    color_fill = '#2a9d8e' # Teal for density curves
    
    # Draw Point Scale at the top
    # Replicate regplot style: solid line with ticks
    header_y = len(top_features) * row_height + 1.2
    ax.plot([0, 100], [header_y, header_y], color=color_lines, lw=1.2)
    for i in range(0, 101, 10):
        # Major ticks
        ax.plot([i, i], [header_y, header_y + 0.25], color=color_lines, lw=1)
        ax.text(i, header_y + 0.4, str(i), ha='center', va='bottom', fontsize=12, color=color_text)
        
        # Minor ticks (midpoints)
        if i < 100:
            mid = i + 5
            ax.plot([mid, mid], [header_y, header_y + 0.15], color=color_lines, lw=0.8)
            
    ax.text(50, header_y + 0.8, 'Points', ha='center', va='bottom', fontsize=16, fontweight='bold', color=color_text)
    
    # Draw Feature Bars with Density Curves
    y_pos = len(top_features) * row_height
    for _, row in top_features.iterrows():
        feat = row['feature']
        coef = row['coef']
        min_v = row['min_val']
        max_v = row['max_val']
        
        # Calculate points for min and max
        p_min = (min_v * coef - row['eff_min']) * scale 
        p_max = (max_v * coef - row['eff_min']) * scale 
        
        # Refined Ticks
        if row['max_val'] - row['min_val'] <= 1: # Binary
            ticks = [0, 1]
        else:
            ticks = np.linspace(row['min_val'], row['max_val'], 6) # More ticks
            
        # Draw base line
        ax.plot([0, row['range'] * scale], [y_pos, y_pos], color=color_lines, lw=1)
        ax.text(-5, y_pos, feat, ha='right', va='center', fontsize=15, color=color_text, fontweight='medium')
        
        for t in ticks:
            # Calculate points for value t
            val_contribution = t * coef
            pts = (val_contribution - row['eff_min']) * scale
            
            # Draw tick
            ax.plot([pts, pts], [y_pos, y_pos + 0.2], color=color_lines, lw=1)
            
            # Format label
            if abs(t) < 0.001: l_str = "0"
            elif abs(t - round(t)) < 0.001: l_str = f"{int(round(t))}"
            elif abs(t) >= 10: l_str = f"{int(round(t))}"
            else: l_str = f"{t:.1f}"
            
            ax.text(pts, y_pos + 0.35, l_str, ha='center', va='bottom', fontsize=11, color=color_text)
        
        # --- Add Density Curve ---
        vals = X[feat]
        # Calculate points for all samples for this feature
        points_dist = (vals * coef - row['eff_min']) * scale
        
        if vals.nunique() > 5: # Continuous: KDE
             try:
                 kde = gaussian_kde(points_dist)
                 # Generate grid within the range of points
                 # slightly extended to cover tails
                 x_min, x_max = points_dist.min(), points_dist.max()
                 margin = (x_max - x_min) * 0.1
                 x_grid = np.linspace(max(0, x_min - margin), min(row['range']*scale, x_max + margin), 100)
                 y_grid = kde(x_grid)
                 
                 # Normalize height: Max height approx 0.5 unit (relative to 0.8 row spacing)
                 if y_grid.max() > 0:
                     y_grid = y_grid / y_grid.max() * 0.5 
                 
                 # Plot filled curve
                 ax.fill_between(x_grid, y_pos + y_grid, y_pos, color=color_fill, alpha=0.3)
                 ax.plot(x_grid, y_pos + y_grid, color=color_fill, lw=1)
             except Exception as e:
                 pass # Skip if KDE fails (e.g. singular matrix)
        else:
            # Discrete: Draw simple histogram-like bars or rug plot?
            # Let's draw bars at each unique value
            # Height proportional to count
            counts = vals.value_counts(normalize=True)
            for v, freq in counts.items():
                pt = (v * coef - row['eff_min']) * scale
                # Height up to 0.5 for 100% freq
                h = freq / counts.max() * 0.5
                ax.fill_between([pt-0.5, pt+0.5], [y_pos+h, y_pos+h], [y_pos, y_pos], color=color_fill, alpha=0.5)
            
        y_pos -= row_height # More spacing

    # --- Total Points Scale and Distribution ---
    footer_y = 0
    
    # Calculate Total Points for ALL samples
    # We need to sum up (X[col] * coef - eff_min[col]) * scale for all used columns
    # We use 'contributions' to get eff_min and coef for all cols
    
    total_points_samples = np.zeros(len(X))
    for _, row in contributions_df.iterrows():
         feat = row['feature']
         coef = row['coef']
         eff_min = row['eff_min']
         
         pts = (X[feat] * coef - eff_min) * scale
         total_points_samples += pts
         
    total_max_points = contributions_df['range'].sum() * scale
    
    # "Shrink method": Map 0-TotalMax to 0-100 width.
    scale_total = 100.0 / total_max_points
    
    ax.plot([0, 100], [footer_y, footer_y], color=color_lines, lw=1.2)
    ax.text(-5, footer_y, 'Total Points', ha='right', va='center', fontsize=15, color=color_text)
    
    # Ticks for Total Points
    step = 20
    if total_max_points > 200: step = 50
    if total_max_points > 500: step = 100
    
    tp_ticks = np.arange(0, total_max_points + step, step)
    for tp in tp_ticks:
        x_pos = tp * scale_total 
        if x_pos > 100: continue 
        ax.plot([x_pos, x_pos], [footer_y, footer_y - 0.25], color=color_lines, lw=1)
        ax.text(x_pos, footer_y - 0.4, str(int(tp)), ha='center', va='top', fontsize=12, color=color_text)

    # Total Points Distribution Curve
    try:
         tp_mapped = total_points_samples * scale_total
         kde_tp = gaussian_kde(tp_mapped)
         x_grid_tp = np.linspace(0, 100, 200)
         y_grid_tp = kde_tp(x_grid_tp)
         # Normalize height
         if y_grid_tp.max() > 0:
             y_grid_tp = y_grid_tp / y_grid_tp.max() * 0.5
         
         ax.fill_between(x_grid_tp, footer_y + y_grid_tp, footer_y, color=color_fill, alpha=0.3)
         ax.plot(x_grid_tp, footer_y + y_grid_tp, color=color_fill, lw=1)
    except:
         pass

    # --- Linear Predictor Scale (Prediction) ---
    pred_y = footer_y - 1.5
    
    # Calculate Prediction Range corresponding to Total Points
    # Prediction = Intercept + Sum(eff_min) + (Total Points / Max Total Points) * (Max Prediction - Min Prediction)?
    # No. 
    # Prediction = Intercept + Sum(eff_min) + (Total Points / scale)
    base_pred = intercept + contributions_df['eff_min'].sum()
    
    ax.plot([0, 100], [pred_y, pred_y], color=color_lines, lw=1.2)
    ax.text(-5, pred_y, 'Linear Predictor', ha='right', va='center', fontsize=15, color=color_text)
    
    # Draw ticks for Linear Predictor
    # We align them to the Total Points ticks
    for tp in tp_ticks:
        x_pos = tp * scale_total
        if x_pos > 100: continue
        
        pred_val = base_pred + (tp / scale)
        
        ax.plot([x_pos, x_pos], [pred_y, pred_y - 0.25], color=color_lines, lw=1)
        
        # Round logic
        if abs(pred_val) < 0.1: p_str = f"{pred_val:.2f}"
        elif abs(pred_val) >= 10: p_str = f"{int(round(pred_val))}"
        else: p_str = f"{pred_val:.1f}"
            
        ax.text(x_pos, pred_y - 0.4, p_str, ha='center', va='top', fontsize=12, color=color_text)


    ax.set_xlim(-15, 115)
    ax.set_ylim(-2.5, len(top_features) * row_height + 2) # Extend bottom for new scale
    ax.axis('off')
    
    plt.title(f"{target_name} Nomogram", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'nomogram_{target_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Nomogram saved to {FIGURES_DIR}")


def main():
    print("Starting visualization generation...")
    
    # 1. Load Data
    try:
        df = create_analysis_dataset()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter valid data
    df = df.dropna(subset=['age', 'industry_simplified', 'partner', 'total_score', 'estimated_votes'])
    
    # 2. Prepare Data
    # Use logic similar to RandomForestAnalyzer.prepare_features
    # We need to manually do it or instantiate the class
    analyzer = RandomForestAnalyzer()
    categorical_cols = ['industry_simplified', 'age_group']
    numerical_cols = ['age', 'week', 'remaining_contestants', 'is_domestic', 'partner_experience']
    
    X = analyzer.prepare_features(df, categorical_cols, numerical_cols)
    
    # Target 1: Judge Score
    y_judge = df['total_score']
    
    # Target 2: Fan Votes (log)
    y_vote = np.log1p(df['estimated_votes'])
    
    # --- Generate for Judge Score ---
    print("\n--- Processing Judge Score ---")
    rf_judge = train_rf_model(X, y_judge)
    plot_shap_analysis(rf_judge, X, "Judge Score")
    plot_static_nomogram(X, y_judge, "Judge Score")

    # --- Generate for Fan Votes ---
    print("\n--- Processing Fan Votes ---")
    rf_vote = train_rf_model(X, y_vote)
    plot_shap_analysis(rf_vote, X, "Log Fan Votes")
    plot_static_nomogram(X, y_vote, "Log Fan Votes")

if __name__ == "__main__":
    main()
