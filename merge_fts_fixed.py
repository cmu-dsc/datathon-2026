"""
Fixed FTS merge - aggregate by country-year before merging.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("MERGING FTS DATA (FIXED) & CREATING FUNDING ANALYSIS")
print("=" * 70)

# ============================================================================
# STEP 1: Load and AGGREGATE FTS data by country-year
# ============================================================================
print("\n--- Loading and Aggregating FTS Data ---")
fts = pd.read_csv('data/fts_requirements_funding_global.csv')

# Skip HXL row
if fts.iloc[0]['countryCode'] == '#country+code':
    fts = fts.iloc[1:].reset_index(drop=True)

# Convert to proper types
fts['year'] = pd.to_numeric(fts['year'], errors='coerce')
fts['requirements'] = pd.to_numeric(fts['requirements'], errors='coerce')
fts['funding'] = pd.to_numeric(fts['funding'], errors='coerce')
fts['percentFunded'] = pd.to_numeric(fts['percentFunded'], errors='coerce')

# Filter to 2020-2025
fts = fts[(fts['year'] >= 2020) & (fts['year'] <= 2025)]
fts = fts.rename(columns={'countryCode': 'ISO3', 'year': 'Year'})

# AGGREGATE by country-year (sum requirements and funding, weighted avg for percentFunded)
fts_agg = fts.groupby(['ISO3', 'Year']).agg({
    'requirements': 'sum',
    'funding': 'sum',
    'name': 'count'  # Count number of plans
}).reset_index()

# Calculate percentFunded from aggregated values
fts_agg['percentFunded'] = (
    fts_agg['funding'] / fts_agg['requirements'].replace(0, np.nan) * 100
).fillna(0)

# Rename columns
fts_agg = fts_agg.rename(columns={
    'requirements': 'FTS_Requirements',
    'funding': 'FTS_Funding',
    'percentFunded': 'FTS_Percent_Funded',
    'name': 'FTS_Plan_Count'
})

# Calculate funding gap
fts_agg['FTS_Funding_Gap'] = fts_agg['FTS_Requirements'] - fts_agg['FTS_Funding']

print(f"FTS aggregated: {len(fts_agg)} unique country-year combinations")
print(f"Countries: {fts_agg['ISO3'].nunique()}")

# ============================================================================
# STEP 2: Load current predictor variables
# ============================================================================
print("\n--- Loading Current Predictor Variables ---")
df = pd.read_csv('person3_predictor_variables.csv')
print(f"Current dataset: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# STEP 3: Merge FTS data (should be 1:1 now)
# ============================================================================
print("\n--- Merging FTS Data ---")
df_merged = df.merge(fts_agg, on=['ISO3', 'Year'], how='left')
print(f"After merge: {len(df_merged)} rows (should be {len(df)})")

if len(df_merged) != len(df):
    print("WARNING: Merge created duplicates! Investigating...")
else:
    print("SUCCESS: No duplicates created")

# Check FTS coverage
fts_coverage = df_merged['FTS_Funding'].notna().sum()
print(f"FTS data coverage: {fts_coverage} / {len(df_merged)} ({100*fts_coverage/len(df_merged):.1f}%)")

# ============================================================================
# STEP 4: Comprehensive Funding Report
# ============================================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE FUNDING DATA REPORT")
print("=" * 70)

print("\n--- Funding Sources Summary ---")
summary = pd.DataFrame({
    'Source': ['CERF Allocations', 'CBPF Budget', 'HRP Requirements', 
               'FTS Requirements', 'FTS Actual Funding', 'FTS Funding Gap'],
    'Total (USD)': [
        df_merged['CERF_Allocation'].sum(),
        df_merged['CBPF_Budget'].sum(),
        df_merged['HRP_Revised_Requirements'].sum(),
        df_merged['FTS_Requirements'].sum(),
        df_merged['FTS_Funding'].sum(),
        df_merged['FTS_Funding_Gap'].sum()
    ],
    'Records with Data': [
        (df_merged['CERF_Allocation'] > 0).sum(),
        (df_merged['CBPF_Budget'] > 0).sum(),
        (df_merged['HRP_Revised_Requirements'] > 0).sum(),
        df_merged['FTS_Requirements'].notna().sum(),
        df_merged['FTS_Funding'].notna().sum(),
        df_merged['FTS_Funding_Gap'].notna().sum()
    ]
})
summary['Coverage %'] = (summary['Records with Data'] / len(df_merged) * 100).round(1)
summary['Total (USD)'] = summary['Total (USD)'].apply(lambda x: f"${x:,.0f}")
print(summary.to_string(index=False))

# ============================================================================
# STEP 5: Create Derived Metrics for Scoring
# ============================================================================
print("\n" + "=" * 70)
print("CREATING SCORING METRICS")
print("=" * 70)

# 1. Funding Coverage Rate
df_merged['Funding_Coverage_Rate'] = df_merged['FTS_Percent_Funded'].fillna(0)

# 2. Funding Per Person In Need
df_merged['Funding_Per_PIN'] = (
    df_merged['FTS_Funding'] / df_merged['Total_In_Need'].replace(0, np.nan)
).fillna(0)

# 3. Funding Per Capita
df_merged['Funding_Per_Capita'] = (
    df_merged['FTS_Funding'] / df_merged['Population'].replace(0, np.nan)
).fillna(0)

# 4. Gap as % of Requirements
df_merged['Gap_Percentage'] = (
    df_merged['FTS_Funding_Gap'] / df_merged['FTS_Requirements'].replace(0, np.nan) * 100
).fillna(0).clip(0, 100)

# ============================================================================
# STEP 6: Implement Scoring System
# ============================================================================
print("\n" + "=" * 70)
print("IMPLEMENTING CRISIS RESPONSE EFFECTIVENESS SCORE")
print("=" * 70)

# Normalize function
def normalize_minmax(series, invert=False):
    """Normalize to 0-100 scale."""
    s = series.fillna(0)
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series([50] * len(s), index=s.index)
    normalized = (s - min_val) / (max_val - min_val) * 100
    if invert:
        normalized = 100 - normalized
    return normalized

# Component scores (all on 0-100 scale)

# Coverage Score: Higher FTS_Percent_Funded = better
df_merged['Score_Coverage'] = df_merged['FTS_Percent_Funded'].clip(0, 150).fillna(0)
# Cap at 100
df_merged['Score_Coverage'] = df_merged['Score_Coverage'].clip(0, 100)

# Efficiency Score: Higher funding per person in need = better (normalized)
funding_per_pin_capped = df_merged['Funding_Per_PIN'].clip(0, df_merged['Funding_Per_PIN'].quantile(0.95))
df_merged['Score_Efficiency'] = normalize_minmax(funding_per_pin_capped)

# Outcome Score: Negative INFORM change = improvement = better
# Scale: -1 to +1 change typical, map to 0-100
df_merged['Score_Outcome'] = (50 - df_merged['INFORM_Change'].fillna(0) * 50).clip(0, 100)

# Gap Severity Score: Higher gap % = worse (invert for scoring)
df_merged['Score_Gap'] = (100 - df_merged['Gap_Percentage'].fillna(50)).clip(0, 100)

# Combined Effectiveness Score (weighted average)
# Outcome-first approach: prioritize actual severity improvement
weights = {'coverage': 0.20, 'efficiency': 0.20, 'outcome': 0.40, 'gap': 0.20}

df_merged['Effectiveness_Score'] = (
    weights['coverage'] * df_merged['Score_Coverage'] +
    weights['efficiency'] * df_merged['Score_Efficiency'] +
    weights['outcome'] * df_merged['Score_Outcome'] +
    weights['gap'] * df_merged['Score_Gap']
)

# For rows without FTS data, use a fallback score based on available data
no_fts_mask = df_merged['FTS_Funding'].isna()
df_merged.loc[no_fts_mask, 'Effectiveness_Score'] = (
    0.5 * df_merged.loc[no_fts_mask, 'Score_Outcome'] +
    0.5 * 30  # Default moderate score
)

# Categorize
def categorize_effectiveness(score):
    if score >= 60:
        return 'Highly Effective'
    elif score >= 45:
        return 'Moderately Effective'
    elif score >= 30:
        return 'Needs Improvement'
    else:
        return 'Critical - Underfunded'

df_merged['Effectiveness_Category'] = df_merged['Effectiveness_Score'].apply(categorize_effectiveness)

# Good crisis flag for model training
df_merged['Is_Good_Crisis'] = df_merged['Effectiveness_Score'] >= 45

print("\n--- Effectiveness Score Distribution ---")
print(df_merged['Effectiveness_Category'].value_counts().to_string())

print(f"\nGood Crises (for model training): {df_merged['Is_Good_Crisis'].sum()} / {len(df_merged)}")

# ============================================================================
# STEP 7: Show Results
# ============================================================================
print("\n" + "=" * 70)
print("TOP PERFORMERS & CRITICAL CASES")
print("=" * 70)

# Top 10 best
print("\n--- Top 10 Best Managed Crises ---")
cols_to_show = ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'FTS_Percent_Funded', 
                'Effectiveness_Score', 'Effectiveness_Category']
top_10 = df_merged[df_merged['FTS_Funding'].notna()].nlargest(10, 'Effectiveness_Score')[cols_to_show]
print(top_10.to_string(index=False))

# Bottom 10 (with FTS data)
print("\n--- Top 10 Critical Underfunded Crises (High Severity) ---")
critical = df_merged[
    (df_merged['FTS_Funding'].notna()) & 
    (df_merged['INFORM_Mean'] >= 3.5)
].nsmallest(10, 'Effectiveness_Score')
cols_critical = ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'FTS_Percent_Funded', 
                 'FTS_Funding_Gap', 'Effectiveness_Score']
print(critical[cols_critical].to_string(index=False))

# ============================================================================
# STEP 8: Save outputs
# ============================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)

# Save complete dataset
df_merged.to_csv('complete_funding_dataset.csv', index=False)
print(f"Saved: complete_funding_dataset.csv ({len(df_merged)} rows, {len(df_merged.columns)} columns)")

# Save effectiveness scores (Person 2 output)
score_cols = ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'INFORM_Change',
              'FTS_Requirements', 'FTS_Funding', 'FTS_Percent_Funded', 'FTS_Funding_Gap',
              'Score_Coverage', 'Score_Efficiency', 'Score_Outcome', 'Score_Gap',
              'Effectiveness_Score', 'Effectiveness_Category', 'Is_Good_Crisis']
df_merged[score_cols].to_csv('person2_effectiveness_scores.csv', index=False)
print("Saved: person2_effectiveness_scores.csv")

# ============================================================================
# STEP 9: Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
DATASET OVERVIEW:
- Total records: {len(df_merged)} country-year combinations
- Total columns: {len(df_merged.columns)}
- Date range: {df_merged['Year'].min():.0f} - {df_merged['Year'].max():.0f}
- Countries: {df_merged['ISO3'].nunique()}

FUNDING DATA COVERAGE:
- FTS Requirements: {df_merged['FTS_Requirements'].notna().sum()} records (${df_merged['FTS_Requirements'].sum():,.0f})
- FTS Actual Funding: {df_merged['FTS_Funding'].notna().sum()} records (${df_merged['FTS_Funding'].sum():,.0f})
- Average % Funded: {df_merged['FTS_Percent_Funded'].mean():.1f}%
- Total Funding Gap: ${df_merged['FTS_Funding_Gap'].sum():,.0f}

EFFECTIVENESS SCORING:
- Highly Effective: {(df_merged['Effectiveness_Category'] == 'Highly Effective').sum()}
- Moderately Effective: {(df_merged['Effectiveness_Category'] == 'Moderately Effective').sum()}
- Needs Improvement: {(df_merged['Effectiveness_Category'] == 'Needs Improvement').sum()}
- Critical - Underfunded: {(df_merged['Effectiveness_Category'] == 'Critical - Underfunded').sum()}

MODEL TRAINING:
- Good crises (Is_Good_Crisis=True): {df_merged['Is_Good_Crisis'].sum()}
- These can be used to train a model predicting "optimal" funding levels

FILES CREATED:
1. complete_funding_dataset.csv - Full dataset with all features + scores
2. person2_effectiveness_scores.csv - Effectiveness scores for each crisis
""")
