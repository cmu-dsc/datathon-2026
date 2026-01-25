"""
Create visualizations for Datathon 2026 presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for plots
import os
os.makedirs('visualizations', exist_ok=True)

print("Loading data...")
predictions = pd.read_csv('person1_all_predictions.csv')
performance = pd.read_csv('person1_model_performance.csv')
importance = pd.read_csv('person1_feature_importance.csv')
effectiveness = pd.read_csv('person2_effectiveness_scores.csv')
complete_data = pd.read_csv('complete_funding_dataset.csv')

# Helper function for currency formatting
def billions(x, pos):
    return f'${x/1e9:.1f}B'

def millions(x, pos):
    return f'${x/1e6:.0f}M'

# ============================================================================
# 1. MODEL PERFORMANCE COMPARISON
# ============================================================================
print("Creating model performance chart...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# R² Score
ax1 = axes[0]
colors = ['#2ecc71', '#3498db', '#9b59b6']
bars1 = ax1.bar(performance['Model'], performance['R2'], color=colors, edgecolor='white', linewidth=2)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Model Accuracy (R²)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
for bar, val in zip(bars1, performance['R2']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.tick_params(axis='x', rotation=15)

# MAE
ax2 = axes[1]
bars2 = ax2.bar(performance['Model'], performance['MAE']/1e6, color=colors, edgecolor='white', linewidth=2)
ax2.set_ylabel('MAE (Millions USD)', fontsize=12)
ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, performance['MAE']/1e6):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'${val:.0f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', rotation=15)

# RMSE
ax3 = axes[2]
bars3 = ax3.bar(performance['Model'], performance['RMSE']/1e6, color=colors, edgecolor='white', linewidth=2)
ax3.set_ylabel('RMSE (Millions USD)', fontsize=12)
ax3.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
for bar, val in zip(bars3, performance['RMSE']/1e6):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'${val:.0f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('visualizations/01_model_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/01_model_performance.png")

# ============================================================================
# 2. FEATURE IMPORTANCE
# ============================================================================
print("Creating feature importance chart...")
fig, ax = plt.subplots(figsize=(10, 8))

top_features = importance.head(12)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))

bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top 12 Features Predicting Funding Needs', fontsize=14, fontweight='bold')

for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{val:.1%}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/02_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/02_feature_importance.png")

# ============================================================================
# 3. FUNDING STATUS DISTRIBUTION
# ============================================================================
print("Creating funding status chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
ax1 = axes[0]
status_counts = predictions['Funding_Status'].value_counts()
colors_status = {'Well Funded': '#27ae60', 'Adequately Funded': '#3498db', 
                 'Underfunded': '#f39c12', 'Severely Underfunded': '#e74c3c',
                 'No Funding Data': '#95a5a6'}
pie_colors = [colors_status.get(s, '#95a5a6') for s in status_counts.index]

wedges, texts, autotexts = ax1.pie(status_counts.values, labels=status_counts.index, 
                                    autopct='%1.1f%%', colors=pie_colors, startangle=90,
                                    explode=[0.05 if 'Under' in s else 0 for s in status_counts.index])
ax1.set_title('Crisis Funding Status Distribution', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

# Bar chart by region
ax2 = axes[1]
region_status = predictions.groupby(['UN_Region', 'Funding_Status']).size().unstack(fill_value=0)
region_status = region_status.reindex(columns=['Well Funded', 'Adequately Funded', 'Underfunded', 
                                                'Severely Underfunded', 'No Funding Data'], fill_value=0)
region_status.plot(kind='bar', stacked=True, ax=ax2, 
                   color=[colors_status[c] for c in region_status.columns], edgecolor='white')
ax2.set_xlabel('UN Region', fontsize=12)
ax2.set_ylabel('Number of Crises', fontsize=12)
ax2.set_title('Funding Status by Region', fontsize=14, fontweight='bold')
ax2.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/03_funding_status.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/03_funding_status.png")

# ============================================================================
# 4. PREDICTED VS ACTUAL FUNDING
# ============================================================================
print("Creating predictions vs actual chart...")
fig, ax = plt.subplots(figsize=(10, 8))

# Filter to rows with actual funding
plot_data = predictions[predictions['Actual_Funding'] > 0].copy()

# Color by funding status
status_colors = {'Well Funded': '#27ae60', 'Adequately Funded': '#3498db', 
                 'Underfunded': '#f39c12', 'Severely Underfunded': '#e74c3c'}

for status, color in status_colors.items():
    mask = plot_data['Funding_Status'] == status
    ax.scatter(plot_data.loc[mask, 'Actual_Funding']/1e9, 
               plot_data.loc[mask, 'Predicted_Funding']/1e9,
               c=color, label=status, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

# Perfect prediction line
max_val = max(plot_data['Actual_Funding'].max(), plot_data['Predicted_Funding'].max()) / 1e9
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction', linewidth=2)

ax.set_xlabel('Actual Funding (Billions USD)', fontsize=12)
ax.set_ylabel('Predicted Funding (Billions USD)', fontsize=12)
ax.set_title('Model Predictions vs Actual Funding', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(0, max_val * 1.05)
ax.set_ylim(0, max_val * 1.05)

plt.tight_layout()
plt.savefig('visualizations/04_predictions_vs_actual.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/04_predictions_vs_actual.png")

# ============================================================================
# 5. TOP UNDERFUNDED CRISES
# ============================================================================
print("Creating underfunded crises chart...")
fig, ax = plt.subplots(figsize=(12, 8))

# Top 15 underfunded with high severity
underfunded = predictions[
    (predictions['Actual_Funding'] > 0) & 
    (predictions['INFORM_Mean'] >= 3.0) &
    (predictions['Funding_Gap'] > 0)
].nlargest(15, 'Funding_Gap').copy()

underfunded['Label'] = underfunded['Country'] + ' (' + underfunded['Year'].astype(int).astype(str) + ')'
underfunded = underfunded.sort_values('Funding_Gap')

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(underfunded)))
bars = ax.barh(range(len(underfunded)), underfunded['Funding_Gap']/1e6, color=colors)
ax.set_yticks(range(len(underfunded)))
ax.set_yticklabels(underfunded['Label'], fontsize=11)
ax.set_xlabel('Funding Gap (Millions USD)', fontsize=12)
ax.set_title('Top 15 Underfunded High-Severity Crises', fontsize=14, fontweight='bold')

for bar, val in zip(bars, underfunded['Funding_Gap']/1e6):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
            f'${val:.0f}M', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/05_underfunded_crises.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/05_underfunded_crises.png")

# ============================================================================
# 6. EFFECTIVENESS SCORE DISTRIBUTION
# ============================================================================
print("Creating effectiveness score chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
ax1 = axes[0]
ax1.hist(effectiveness['Effectiveness_Score'].dropna(), bins=25, color='#3498db', 
         edgecolor='white', alpha=0.8)
ax1.axvline(x=45, color='#e74c3c', linestyle='--', linewidth=2, label='Good Crisis Threshold (45)')
ax1.axvline(x=effectiveness['Effectiveness_Score'].mean(), color='#27ae60', 
            linestyle='-', linewidth=2, label=f'Mean ({effectiveness["Effectiveness_Score"].mean():.1f})')
ax1.set_xlabel('Effectiveness Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Crisis Response Effectiveness', fontsize=14, fontweight='bold')
ax1.legend()

# Category breakdown
ax2 = axes[1]
cat_counts = effectiveness['Effectiveness_Category'].value_counts()
cat_order = ['Highly Effective', 'Moderately Effective', 'Needs Improvement', 'Critical - Underfunded']
cat_counts = cat_counts.reindex(cat_order)
cat_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']

bars = ax2.bar(range(len(cat_counts)), cat_counts.values, color=cat_colors, edgecolor='white', linewidth=2)
ax2.set_xticks(range(len(cat_counts)))
ax2.set_xticklabels(cat_counts.index, rotation=15, ha='right', fontsize=10)
ax2.set_ylabel('Number of Crises', fontsize=12)
ax2.set_title('Crisis Response Effectiveness Categories', fontsize=14, fontweight='bold')

for bar, val in zip(bars, cat_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/06_effectiveness_scores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/06_effectiveness_scores.png")

# ============================================================================
# 7. FUNDING TRENDS BY YEAR
# ============================================================================
print("Creating funding trends chart...")
fig, ax = plt.subplots(figsize=(12, 6))

# Aggregate by year
yearly = complete_data.groupby('Year').agg({
    'FTS_Funding': 'sum',
    'FTS_Requirements': 'sum',
    'FTS_Funding_Gap': 'sum'
}).dropna()

x = yearly.index.astype(int)
width = 0.35

bars1 = ax.bar(x - width/2, yearly['FTS_Requirements']/1e9, width, label='Requirements', 
               color='#3498db', edgecolor='white')
bars2 = ax.bar(x + width/2, yearly['FTS_Funding']/1e9, width, label='Actual Funding', 
               color='#27ae60', edgecolor='white')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Amount (Billions USD)', fontsize=12)
ax.set_title('Humanitarian Funding: Requirements vs Reality', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.legend()

# Add gap annotation
for i, (req, fund) in enumerate(zip(yearly['FTS_Requirements']/1e9, yearly['FTS_Funding']/1e9)):
    gap = req - fund
    pct = (fund/req)*100 if req > 0 else 0
    ax.annotate(f'{pct:.0f}%\nfunded', xy=(x[i], fund), xytext=(x[i], fund + 3),
                ha='center', fontsize=9, color='#27ae60', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/07_funding_trends.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/07_funding_trends.png")

# ============================================================================
# 8. INFORM SEVERITY VS FUNDING
# ============================================================================
print("Creating severity vs funding chart...")
fig, ax = plt.subplots(figsize=(10, 8))

plot_data = complete_data[complete_data['FTS_Funding'] > 0].copy()

scatter = ax.scatter(plot_data['INFORM_Mean'], plot_data['FTS_Funding']/1e9,
                     c=plot_data['FTS_Percent_Funded'], cmap='RdYlGn', 
                     s=80, alpha=0.7, edgecolors='white', linewidth=0.5,
                     vmin=0, vmax=100)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('% Funded', fontsize=11)

ax.set_xlabel('INFORM Severity Index', fontsize=12)
ax.set_ylabel('Total Funding (Billions USD)', fontsize=12)
ax.set_title('Crisis Severity vs Funding Received', fontsize=14, fontweight='bold')

# Add trend line (handle NaN values)
try:
    valid_mask = plot_data['INFORM_Mean'].notna() & plot_data['FTS_Funding'].notna()
    x_valid = plot_data.loc[valid_mask, 'INFORM_Mean'].values
    y_valid = (plot_data.loc[valid_mask, 'FTS_Funding']/1e9).values
    z = np.polyfit(x_valid, y_valid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2, label='Trend')
    ax.legend()
except:
    pass  # Skip trend line if it fails

plt.tight_layout()
plt.savefig('visualizations/08_severity_vs_funding.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: visualizations/08_severity_vs_funding.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("VISUALIZATIONS COMPLETE")
print("=" * 60)
print(f"""
Created 8 visualizations in the 'visualizations/' folder:

1. 01_model_performance.png    - Model accuracy comparison (R², MAE, RMSE)
2. 02_feature_importance.png   - Top predictive features
3. 03_funding_status.png       - Funding status distribution & by region
4. 04_predictions_vs_actual.png - Model predictions scatter plot
5. 05_underfunded_crises.png   - Top underfunded high-severity crises
6. 06_effectiveness_scores.png - Effectiveness score distribution
7. 07_funding_trends.png       - Yearly funding trends (requirements vs actual)
8. 08_severity_vs_funding.png  - Crisis severity vs funding received

These can be used directly in your presentation!
""")
