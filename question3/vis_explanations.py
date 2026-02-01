
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
import matplotlib.image as mpimg

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


def combine_two_pngs_side_by_side(
    left_png_path: str,
    right_png_path: str,
    out_png_path: str,
    left_title: str,
    right_title: str,
    figure_title: str | None = None,
    bg_color: str = '#F8F9FA',
):
    """Combine two existing PNG files into one (left/right) and save."""

    if not (os.path.exists(left_png_path) and os.path.exists(right_png_path)):
        print(f"[combine] Missing input files: {left_png_path} or {right_png_path}")
        return

    img_left = mpimg.imread(left_png_path)
    img_right = mpimg.imread(right_png_path)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    fig.patch.set_facecolor(bg_color)
    for ax in axes:
        ax.set_facecolor(bg_color)

    axes[0].imshow(img_left)
    axes[0].axis('off')
    axes[0].set_title(left_title, fontsize=16, fontweight='bold', color='#264653', pad=10)

    axes[1].imshow(img_right)
    axes[1].axis('off')
    axes[1].set_title(right_title, fontsize=16, fontweight='bold', color='#264653', pad=10)

    if figure_title:
        fig.suptitle(figure_title, fontsize=18, fontweight='bold', color='#264653')

    fig.savefig(out_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[combine] Saved: {out_png_path}")

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
    color_points = '#264653' # Same as lines for rug plot
    
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
    idx = 0
    for _, row in top_features.iterrows():
        # Zebra Striping: Alternating background for rows
        if idx % 2 == 0:
            # Draw a subtle rectangle behind the entire row
            # x from -15 to 115 (plot limits), y from y_pos - 0.2 to y_pos + row_height - 0.2
            rect = plt.Rectangle((-15, y_pos - 0.1), 130, row_height, 
                                 edgecolor='none', facecolor='#e9ecef', alpha=0.3, zorder=-1)
            ax.add_patch(rect)
        idx += 1
        
        feat = row['feature']
        coef = row['coef']
        min_v = row['min_val']
        max_v = row['max_val']
        
        # Calculate points for min and max
        p_min = (min_v * coef - row['eff_min']) * scale 
        p_max = (max_v * coef - row['eff_min']) * scale 
        
        # Refined Ticks
        # ... (rest of the code)
        if row['max_val'] - row['min_val'] <= 1: # Binary
            ticks = [0, 1]
        else:
            ticks = np.linspace(row['min_val'], row['max_val'], 6) # More ticks
            
        # Draw base line
        ax.plot([0, row['range'] * scale], [y_pos, y_pos], color=color_lines, lw=1.2) # Slightly thicker base line
        ax.text(-5, y_pos, feat, ha='right', va='center', fontsize=15, color=color_text, fontweight='medium')
        
        for t in ticks:
            # Calculate points for value t
            val_contribution = t * coef
            pts = (val_contribution - row['eff_min']) * scale
            
            # Draw tick
            ax.plot([pts, pts], [y_pos, y_pos + 0.2], color=color_lines, lw=1.2)
            
            # Format label
            if abs(t) < 0.001: l_str = "0"
            elif abs(t - round(t)) < 0.001: l_str = f"{int(round(t))}"
            elif abs(t) >= 10: l_str = f"{int(round(t))}"
            else: l_str = f"{t:.1f}"
            
            ax.text(pts, y_pos + 0.35, l_str, ha='center', va='bottom', fontsize=11, color=color_text)
        
        # --- Add Density Curve with Gradient-like Effect and Rug Plot ---
        vals = X[feat]
        # Calculate points for all samples for this feature
        points_dist = (vals * coef - row['eff_min']) * scale
        
        # Rug Plot (Bottom of the axis)
        # Add small vertical ticks for actual data points
        ax.scatter(points_dist, [y_pos - 0.05] * len(points_dist), 
                   marker='|', color=color_points, alpha=0.3, s=30)

        if vals.nunique() > 5: # Continuous: KDE
             try:
                 kde = gaussian_kde(points_dist)
                 # Generate grid within the range of points
                 # slightly extended to cover tails
                 x_min, x_max = points_dist.min(), points_dist.max()
                 margin = (x_max - x_min) * 0.1
                 x_grid_extended = np.linspace(max(0, x_min - margin), min(row['range']*scale, x_max + margin), 200)
                 y_grid = kde(x_grid_extended)
                 
                 # Normalize height: Max height approx 0.5 unit
                 if y_grid.max() > 0:
                     y_grid = y_grid / y_grid.max() * 0.5 
                 
                 # Plot filled curve with slightly darker edge
                 ax.fill_between(x_grid_extended, y_pos + y_grid, y_pos, color=color_fill, alpha=0.4)
                 ax.plot(x_grid_extended, y_pos + y_grid, color=color_fill, lw=1.5)
             except Exception as e:
                 pass # Skip if KDE fails (e.g. singular matrix)
        else:
            # Discrete: Draw simple histogram-like bars
            counts = vals.value_counts(normalize=True)
            for v, freq in counts.items():
                pt = (v * coef - row['eff_min']) * scale
                # Height up to 0.5 for 100% freq
                h = freq / counts.max() * 0.5
                ax.fill_between([pt-0.5, pt+0.5], [y_pos+h, y_pos+h], [y_pos, y_pos], color=color_fill, alpha=0.5, edgecolor=color_fill)
            
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


def plot_residual_case_order(model, X, y, target_name="Target"):
    """
    Plot Residuals vs Case Order to check for independence/autocorrelation.
    """
    print(f"Generating Residual Case Order Plot for {target_name}...")
    
    # Get predictions
    if hasattr(model, 'predict'):
        preds = model.predict(X)
    else:
        # Fallback if model is not a standard sklearn estimator
        print(f"Model {type(model)} does not have predict method.")
        return

    residuals = y - preds
    
    plt.figure(figsize=(12, 6))
    
    # Background
    bg_color = '#F8F9FA'
    plt.gcf().patch.set_facecolor(bg_color)
    plt.gca().set_facecolor(bg_color)
    
    # Scatter plot
    plt.scatter(range(len(residuals)), residuals, color=MCM_COLORS[0], alpha=0.5, s=20, label='Residuals')
    
    # Zero line
    plt.axhline(0, color=MCM_COLORS[4], linestyle='--', lw=2, label='Zero Mean')
    
    # Smoothed trend line (rolling mean)
    window = int(len(residuals) * 0.05) if len(residuals) > 100 else 10
    if window < 2: window = 2
    rolling_mean = pd.Series(residuals).rolling(window=window, center=True).mean()
    plt.plot(rolling_mean, color=MCM_COLORS[2], lw=2.5, label=f'Moving Avg (w={window})')
    
    plt.title(f"{target_name} - Residual Case Order Plot", fontsize=18, fontweight='bold', color='#264653', pad=15)
    plt.xlabel("Case Order (Observation Index)", fontsize=15, color='#264653')
    plt.ylabel("Residuals (Observed - Predicted)", fontsize=15, color='#264653')
    
    # Ticks styling
    plt.tick_params(colors='#264653', which='both')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', color='#264653')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'residual_case_order_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual plot saved to {save_path}")

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
    plot_shap_analysis(rf_judge, X, "judge_score")
    plot_static_nomogram(X, y_judge, "judge_score")
    plot_residual_case_order(rf_judge, X, y_judge, "judge_score")

    # --- Generate for Fan Votes ---
    print("\n--- Processing Fan Votes ---")
    rf_vote = train_rf_model(X, y_vote)
    plot_shap_analysis(rf_vote, X, "fan_votes")
    plot_static_nomogram(X, y_vote, "fan_votes")
    plot_residual_case_order(rf_vote, X, y_vote, "fan_votes")

    # --- Combine Judge vs Fan plots by type ---
    print("\n--- Combining figures (Judge vs Fan) ---")

    # Residual (keep only one combined file)
    residual_left = os.path.join(FIGURES_DIR, 'residual_case_order_judge_score.png')
    residual_right = os.path.join(FIGURES_DIR, 'residual_case_order_fan_votes.png')
    residual_out = os.path.join(FIGURES_DIR, 'residual_case_order_combined.png')
    combine_two_pngs_side_by_side(
        residual_left,
        residual_right,
        residual_out,
        left_title='Judge Score',
        right_title='Fan Votes (log)',
        figure_title='Residual Case Order Plot',
    )
    for p in (residual_left, residual_right):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # SHAP Summary
    shap_sum_left = os.path.join(FIGURES_DIR, 'shap_summary_judge_score.png')
    shap_sum_right = os.path.join(FIGURES_DIR, 'shap_summary_fan_votes.png')
    shap_sum_out = os.path.join(FIGURES_DIR, 'shap_summary_combined.png')
    combine_two_pngs_side_by_side(
        shap_sum_left,
        shap_sum_right,
        shap_sum_out,
        left_title='Judge Score',
        right_title='Fan Votes (log)',
        figure_title='SHAP Summary (Beeswarm)',
    )
    for p in (shap_sum_left, shap_sum_right):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # SHAP Importance (Bar)
    shap_imp_left = os.path.join(FIGURES_DIR, 'shap_importance_judge_score.png')
    shap_imp_right = os.path.join(FIGURES_DIR, 'shap_importance_fan_votes.png')
    shap_imp_out = os.path.join(FIGURES_DIR, 'shap_importance_combined.png')
    combine_two_pngs_side_by_side(
        shap_imp_left,
        shap_imp_right,
        shap_imp_out,
        left_title='Judge Score',
        right_title='Fan Votes (log)',
        figure_title='SHAP Feature Importance',
    )
    for p in (shap_imp_left, shap_imp_right):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # Nomogram
    nom_left = os.path.join(FIGURES_DIR, 'nomogram_judge_score.png')
    nom_right = os.path.join(FIGURES_DIR, 'nomogram_fan_votes.png')
    nom_out = os.path.join(FIGURES_DIR, 'nomogram_combined.png')
    combine_two_pngs_side_by_side(
        nom_left,
        nom_right,
        nom_out,
        left_title='Judge Score',
        right_title='Fan Votes (log)',
        figure_title='Nomogram (Side-by-side)',
    )
    for p in (nom_left, nom_right):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

if __name__ == "__main__":
    main()
