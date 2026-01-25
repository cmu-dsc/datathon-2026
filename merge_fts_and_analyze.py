"""
Merge FTS data into predictor variables and create comprehensive funding analysis.
This also creates the foundation for Person 2's effectiveness scoring system.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("MERGING FTS DATA & CREATING FUNDING ANALYSIS")
print("=" * 70)

# ============================================================================
# STEP 1: Load and prepare FTS data
# ============================================================================
print("\n--- Loading FTS Data ---")
fts = pd.read_csv('data/fts_requirements_funding_global.csv')

# Skip HXL row
if fts.iloc[0]['countryCode'] == '#country+code':
    fts = fts.iloc[1:].reset_index(drop=True)

# Convert to proper types
fts['year'] = pd.to_numeric(fts['year'], errors='coerce')
fts['requirements'] = pd.to_numeric(fts['requirements'], errors='coerce')
fts['funding'] = pd.to_numeric(fts['funding'], errors='coerce')
fts['percentFunded'] = pd.to_numeric(fts['percentFunded'], errors='coerce')

# Rename and select columns
fts = fts.rename(columns={
    'countryCode': 'ISO3',
    'year': 'Year',
    'requirements': 'FTS_Requirements',
    'funding': 'FTS_Funding',
    'percentFunded': 'FTS_Percent_Funded'
})

# Calculate funding gap
fts['FTS_Funding_Gap'] = fts['FTS_Requirements'] - fts['FTS_Funding']

# Keep only relevant columns
fts_merge = fts[['ISO3', 'Year', 'FTS_Requirements', 'FTS_Funding', 
                  'FTS_Percent_Funded', 'FTS_Funding_Gap']].copy()

# Filter to 2020-2025
fts_merge = fts_merge[(fts_merge['Year'] >= 2020) & (fts_merge['Year'] <= 2025)]

print(f"FTS data prepared: {len(fts_merge)} records")

# ============================================================================
# STEP 2: Load current predictor variables
# ============================================================================
print("\n--- Loading Current Predictor Variables ---")
df = pd.read_csv('person3_predictor_variables.csv')
print(f"Current dataset: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# STEP 3: Merge FTS data
# ============================================================================
print("\n--- Merging FTS Data ---")
df_merged = df.merge(fts_merge, on=['ISO3', 'Year'], how='left')
print(f"After merge: {len(df_merged)} rows, {len(df_merged.columns)} columns")

# Check FTS coverage
fts_coverage = df_merged['FTS_Funding'].notna().sum()
print(f"FTS data coverage: {fts_coverage} / {len(df_merged)} ({100*fts_coverage/len(df_merged):.1f}%)")

# ============================================================================
# STEP 4: Create comprehensive funding report
# ============================================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE FUNDING DATA REPORT")
print("=" * 70)

funding_columns = {
    'CERF_Allocation': 'CERF emergency fund allocations',
    'CERF_Project_Count': 'Number of CERF projects',
    'CBPF_Budget': 'Country-Based Pooled Fund budget',
    'CBPF_Project_Count': 'Number of CBPF projects',
    'CBPF_Total_Beneficiaries': 'Total beneficiaries from CBPF projects',
    'HRP_Original_Requirements': 'Original HRP funding requirements',
    'HRP_Revised_Requirements': 'Revised HRP funding requirements',
    'HRP_Plan_Count': 'Number of HRP plans',
    'Total_Funding': 'Sum of CERF + CBPF + HRP (our original total)',
    'FTS_Requirements': 'FTS: Total funding requested',
    'FTS_Funding': 'FTS: Actual funding received',
    'FTS_Percent_Funded': 'FTS: Percentage of requirements funded',
    'FTS_Funding_Gap': 'FTS: Unmet funding needs (requirements - funding)'
}

print("\n--- Available Funding Columns ---")
for col, desc in funding_columns.items():
    if col in df_merged.columns:
        non_null = df_merged[col].notna().sum()
        non_zero = (df_merged[col] > 0).sum() if df_merged[col].dtype in ['float64', 'int64'] else 'N/A'
        total = df_merged[col].sum() if df_merged[col].dtype in ['float64', 'int64'] else 'N/A'
        print(f"\n  {col}:")
        print(f"    Description: {desc}")
        print(f"    Non-null: {non_null} / {len(df_merged)}")
        print(f"    Non-zero: {non_zero}")
        if isinstance(total, (int, float)) and not pd.isna(total):
            print(f"    Total: ${total:,.0f}")

print("\n" + "=" * 70)
print("FUNDING DATA SUMMARY BY SOURCE")
print("=" * 70)

summary = {
    'Source': ['CERF', 'CBPF', 'HRP Requirements', 'FTS Requirements', 'FTS Actual Funding'],
    'Total Amount': [
        df_merged['CERF_Allocation'].sum(),
        df_merged['CBPF_Budget'].sum(),
        df_merged['HRP_Revised_Requirements'].sum(),
        df_merged['FTS_Requirements'].sum(),
        df_merged['FTS_Funding'].sum()
    ],
    'Countries with Data': [
        (df_merged['CERF_Allocation'] > 0).sum(),
        (df_merged['CBPF_Budget'] > 0).sum(),
        (df_merged['HRP_Revised_Requirements'] > 0).sum(),
        df_merged['FTS_Requirements'].notna().sum(),
        df_merged['FTS_Funding'].notna().sum()
    ],
    'Coverage %': [
        (df_merged['CERF_Allocation'] > 0).sum() / len(df_merged) * 100,
        (df_merged['CBPF_Budget'] > 0).sum() / len(df_merged) * 100,
        (df_merged['HRP_Revised_Requirements'] > 0).sum() / len(df_merged) * 100,
        df_merged['FTS_Requirements'].notna().sum() / len(df_merged) * 100,
        df_merged['FTS_Funding'].notna().sum() / len(df_merged) * 100
    ]
}

summary_df = pd.DataFrame(summary)
summary_df['Total Amount'] = summary_df['Total Amount'].apply(lambda x: f"${x:,.0f}")
summary_df['Coverage %'] = summary_df['Coverage %'].apply(lambda x: f"{x:.1f}%")
print(summary_df.to_string(index=False))

# ============================================================================
# STEP 5: Create Derived Funding Metrics for Scoring
# ============================================================================
print("\n" + "=" * 70)
print("CREATING DERIVED FUNDING METRICS FOR SCORING")
print("=" * 70)

# 1. Funding Coverage Rate (using FTS as ground truth)
df_merged['Funding_Coverage_Rate'] = df_merged['FTS_Percent_Funded'].fillna(
    # Fallback: use CERF+CBPF as % of HRP requirements
    (df_merged['CERF_Allocation'] + df_merged['CBPF_Budget']) / 
    df_merged['HRP_Revised_Requirements'].replace(0, np.nan) * 100
).fillna(0)

# 2. Funding Per Capita
df_merged['FTS_Funding_Per_Capita'] = (
    df_merged['FTS_Funding'] / df_merged['Population'].replace(0, np.nan)
).fillna(0)

# 3. Funding Per Person In Need (efficiency metric)
df_merged['FTS_Funding_Per_PIN'] = (
    df_merged['FTS_Funding'] / df_merged['Total_In_Need'].replace(0, np.nan)
).fillna(0)

# 4. Requirements Per Severity Point
df_merged['Requirements_Per_Severity'] = (
    df_merged['FTS_Requirements'] / df_merged['INFORM_Mean'].replace(0, np.nan)
).fillna(0)

# 5. Funding Adequacy Score (funding relative to severity)
df_merged['Funding_Adequacy_Score'] = (
    df_merged['FTS_Funding'] / 
    (df_merged['INFORM_Mean'].replace(0, np.nan) * df_merged['Population'].replace(0, np.nan))
).fillna(0)

# 6. Gap Severity (funding gap normalized by severity)
df_merged['Gap_Severity_Ratio'] = (
    df_merged['FTS_Funding_Gap'] / 
    (df_merged['INFORM_Mean'].replace(0, np.nan) * 1e9)  # Normalize to billions per severity point
).fillna(0)

print("Created derived funding metrics:")
print("  - Funding_Coverage_Rate: % of requirements met")
print("  - FTS_Funding_Per_Capita: Funding per person in country")
print("  - FTS_Funding_Per_PIN: Funding per person in need")
print("  - Requirements_Per_Severity: Requirements relative to crisis severity")
print("  - Funding_Adequacy_Score: Funding adequacy relative to need")
print("  - Gap_Severity_Ratio: Funding gap relative to severity")

# ============================================================================
# STEP 6: Scoring/Evaluation System Analysis
# ============================================================================
print("\n" + "=" * 70)
print("SCORING/EVALUATION SYSTEM ANALYSIS")
print("=" * 70)

print("""
Based on the available funding data, here are potential scoring approaches:

===============================================================================
OPTION 1: FUNDING EFFECTIVENESS SCORE
===============================================================================
Measures how well funding translates to humanitarian outcomes.

Formula: Effectiveness = (Coverage_Rate * INFORM_Improvement) / 100

Components:
  - Coverage_Rate: FTS_Percent_Funded (% of requirements met)
  - INFORM_Improvement: Negative of INFORM_Change (decrease = good)
  
Interpretation:
  - High score = Good funding coverage AND crisis improvement
  - Low score = Poor coverage OR crisis worsening

===============================================================================
OPTION 2: FUNDING EFFICIENCY SCORE  
===============================================================================
Measures how efficiently funds are used per person in need.

Formula: Efficiency = FTS_Funding_Per_PIN / Regional_Median_Funding_Per_PIN

Components:
  - FTS_Funding_Per_PIN: Funding per person in need
  - Normalized against regional medians
  
Interpretation:
  - Score > 1 = Above average funding per person in need
  - Score < 1 = Below average (potentially underfunded)

===============================================================================
OPTION 3: FUNDING GAP SEVERITY SCORE
===============================================================================
Identifies crises with critical funding gaps relative to severity.

Formula: Gap_Severity = (FTS_Funding_Gap / FTS_Requirements) * INFORM_Mean

Components:
  - Funding gap as % of requirements
  - Weighted by crisis severity
  
Interpretation:
  - High score = Large gap + High severity (CRITICAL)
  - Low score = Small gap OR Low severity

===============================================================================
OPTION 4: CRISIS RESPONSE ADEQUACY SCORE (Recommended)
===============================================================================
Comprehensive score combining multiple factors.

Formula: 
  Adequacy = w1*Coverage + w2*Efficiency + w3*Outcome - w4*Gap_Severity

Where:
  - Coverage: FTS_Percent_Funded (normalized 0-1)
  - Efficiency: Funding_Per_PIN relative to need
  - Outcome: INFORM improvement over time
  - Gap_Severity: Unmet need weighted by severity
  - w1-w4: Weights (sum to 1)

Suggested weights:
  - w1 = 0.30 (Coverage)
  - w2 = 0.25 (Efficiency)  
  - w3 = 0.25 (Outcome)
  - w4 = 0.20 (Gap penalty)

""")

# ============================================================================
# STEP 7: Implement Recommended Scoring System
# ============================================================================
print("=" * 70)
print("IMPLEMENTING CRISIS RESPONSE ADEQUACY SCORE")
print("=" * 70)

# Normalize components to 0-1 scale
def normalize_0_1(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series))
    return (series - min_val) / (max_val - min_val)

# Coverage component (higher is better)
df_merged['Score_Coverage'] = normalize_0_1(
    df_merged['FTS_Percent_Funded'].fillna(0).clip(0, 100)
)

# Efficiency component (higher is better)
df_merged['Score_Efficiency'] = normalize_0_1(
    df_merged['FTS_Funding_Per_PIN'].fillna(0).clip(0, df_merged['FTS_Funding_Per_PIN'].quantile(0.95))
)

# Outcome component (improvement is better - negative change is good)
df_merged['Score_Outcome'] = normalize_0_1(
    -df_merged['INFORM_Change'].fillna(0)  # Negative because decrease is good
)

# Gap severity component (lower is better, so we invert)
gap_severity = (
    df_merged['FTS_Funding_Gap'].fillna(0) / 
    df_merged['FTS_Requirements'].replace(0, np.nan).fillna(1) * 
    df_merged['INFORM_Mean'].fillna(0)
)
df_merged['Score_Gap_Penalty'] = normalize_0_1(gap_severity)

# Calculate final adequacy score
w1, w2, w3, w4 = 0.30, 0.25, 0.25, 0.20
df_merged['Crisis_Response_Adequacy_Score'] = (
    w1 * df_merged['Score_Coverage'] +
    w2 * df_merged['Score_Efficiency'] +
    w3 * df_merged['Score_Outcome'] -
    w4 * df_merged['Score_Gap_Penalty']
).clip(0, 1) * 100  # Scale to 0-100

# Categorize into effectiveness tiers
def categorize_adequacy(score):
    if pd.isna(score):
        return 'Insufficient Data'
    elif score >= 70:
        return 'Highly Effective'
    elif score >= 50:
        return 'Moderately Effective'
    elif score >= 30:
        return 'Needs Improvement'
    else:
        return 'Critical - Underfunded'

df_merged['Response_Effectiveness_Category'] = df_merged['Crisis_Response_Adequacy_Score'].apply(categorize_adequacy)

# Identify "Good Crises" for model training (Person 2's requirement)
df_merged['Is_Good_Crisis'] = df_merged['Crisis_Response_Adequacy_Score'] >= 50

print("\nScoring Distribution:")
print(df_merged['Response_Effectiveness_Category'].value_counts().to_string())

print(f"\nGood Crises (for model training): {df_merged['Is_Good_Crisis'].sum()} / {len(df_merged)}")

# Show top and bottom performers
print("\n--- Top 10 Best Funded/Managed Crises ---")
top_10 = df_merged.nlargest(10, 'Crisis_Response_Adequacy_Score')[
    ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'FTS_Percent_Funded', 
     'Crisis_Response_Adequacy_Score', 'Response_Effectiveness_Category']
]
print(top_10.to_string(index=False))

print("\n--- Top 10 Critical Underfunded Crises ---")
# Filter to those with data
with_data = df_merged[df_merged['FTS_Funding'].notna()]
bottom_10 = with_data.nsmallest(10, 'Crisis_Response_Adequacy_Score')[
    ['ISO3', 'Year', 'Country', 'INFORM_Mean', 'FTS_Percent_Funded', 
     'FTS_Funding_Gap', 'Crisis_Response_Adequacy_Score', 'Response_Effectiveness_Category']
]
print(bottom_10.to_string(index=False))

# ============================================================================
# STEP 8: Save outputs
# ============================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)

# Save complete merged dataset
df_merged.to_csv('complete_funding_dataset.csv', index=False)
print("Saved: complete_funding_dataset.csv")

# Save scoring summary
scoring_summary = df_merged[[
    'ISO3', 'Year', 'Country', 'INFORM_Mean', 'INFORM_Change',
    'FTS_Requirements', 'FTS_Funding', 'FTS_Percent_Funded', 'FTS_Funding_Gap',
    'Score_Coverage', 'Score_Efficiency', 'Score_Outcome', 'Score_Gap_Penalty',
    'Crisis_Response_Adequacy_Score', 'Response_Effectiveness_Category', 'Is_Good_Crisis'
]].copy()
scoring_summary.to_csv('person2_effectiveness_scores.csv', index=False)
print("Saved: person2_effectiveness_scores.csv")

# Create funding data dictionary
funding_dict = []
for col in df_merged.columns:
    if 'fund' in col.lower() or 'fts' in col.lower() or 'score' in col.lower() or 'cerf' in col.lower() or 'cbpf' in col.lower() or 'hrp' in col.lower():
        funding_dict.append({
            'Variable': col,
            'Type': str(df_merged[col].dtype),
            'Non_Null': df_merged[col].notna().sum(),
            'Description': funding_columns.get(col, 'Derived metric')
        })

pd.DataFrame(funding_dict).to_csv('funding_data_dictionary.csv', index=False)
print("Saved: funding_data_dictionary.csv")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)

print(f"""
SUMMARY:
- Complete dataset: {len(df_merged)} rows, {len(df_merged.columns)} columns
- FTS data coverage: {fts_coverage} country-years ({100*fts_coverage/len(df_merged):.1f}%)
- Good crises for model training: {df_merged['Is_Good_Crisis'].sum()}
- Critical underfunded crises: {(df_merged['Response_Effectiveness_Category'] == 'Critical - Underfunded').sum()}

KEY FILES CREATED:
1. complete_funding_dataset.csv - Full merged dataset with all funding + scores
2. person2_effectiveness_scores.csv - Effectiveness scores for each country-year
3. funding_data_dictionary.csv - Documentation of funding variables
""")
