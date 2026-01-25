"""Analyze FTS funding data and compare with existing datasets."""
import pandas as pd
import numpy as np

# Load FTS data and skip HXL row
fts = pd.read_csv('data/fts_requirements_funding_global.csv')
if fts.iloc[0]['countryCode'] == '#country+code':
    fts = fts.iloc[1:].reset_index(drop=True)

# Convert to numeric
fts['year'] = pd.to_numeric(fts['year'], errors='coerce')
fts['requirements'] = pd.to_numeric(fts['requirements'], errors='coerce')
fts['funding'] = pd.to_numeric(fts['funding'], errors='coerce')
fts['percentFunded'] = pd.to_numeric(fts['percentFunded'], errors='coerce')

# Rename for consistency
fts = fts.rename(columns={'countryCode': 'ISO3'})

print('=' * 60)
print('FTS GLOBAL REQUIREMENTS & FUNDING ANALYSIS')
print('=' * 60)

# Filter to relevant years (2020-2025 to match our data)
fts_recent = fts[(fts['year'] >= 2020) & (fts['year'] <= 2025)]
print(f"Records 2020-2025: {len(fts_recent)}")
print(f"Countries: {fts_recent['ISO3'].nunique()}")

print("\n--- Funding Totals (2020-2025) ---")
print(f"Total Requirements: ${fts_recent['requirements'].sum():,.0f}")
print(f"Total Funding Received: ${fts_recent['funding'].sum():,.0f}")
print(f"Average % Funded: {fts_recent['percentFunded'].mean():.1f}%")
print(f"Funding Gap: ${fts_recent['requirements'].sum() - fts_recent['funding'].sum():,.0f}")

print("\n--- By Year ---")
yearly = fts_recent.groupby('year').agg({
    'ISO3': 'nunique',
    'requirements': 'sum',
    'funding': 'sum',
    'percentFunded': 'mean'
}).round(1)
yearly.columns = ['Countries', 'Requirements_USD', 'Funding_USD', 'Avg_Pct_Funded']
print(yearly.to_string())

print("\n--- Top 10 Countries by Funding Received (2020-2025) ---")
country_funding = fts_recent.groupby('ISO3').agg({
    'requirements': 'sum',
    'funding': 'sum',
    'percentFunded': 'mean'
}).sort_values('funding', ascending=False).head(10)
country_funding['funding_gap'] = country_funding['requirements'] - country_funding['funding']
print(country_funding.to_string())

# Load our current merged data for comparison
merged = pd.read_csv('person3_predictor_variables.csv')

print("\n" + "=" * 60)
print("COMPARISON: FTS vs OUR CURRENT FUNDING DATA")
print("=" * 60)

# Compare country coverage
fts_countries = set(fts_recent['ISO3'].unique())
merged_countries = set(merged['ISO3'].unique())
print(f"FTS countries (2020-2025): {len(fts_countries)}")
print(f"Our merged data countries: {len(merged_countries)}")
print(f"Overlap: {len(fts_countries & merged_countries)}")
print(f"In FTS but not our data: {len(fts_countries - merged_countries)}")
print(f"In our data but not FTS: {len(merged_countries - fts_countries)}")

print("\n--- Funding Totals Comparison ---")
our_cerf = merged['CERF_Allocation'].sum()
our_cbpf = merged['CBPF_Budget'].sum()
our_hrp = merged['HRP_Revised_Requirements'].sum()
our_total = merged['Total_Funding'].sum()
fts_total = fts_recent['funding'].sum()
fts_requirements = fts_recent['requirements'].sum()

print(f"Our CERF Allocations:     ${our_cerf:>20,.0f}")
print(f"Our CBPF Budget:          ${our_cbpf:>20,.0f}")
print(f"Our HRP Requirements:     ${our_hrp:>20,.0f}")
print(f"Our Total:                ${our_total:>20,.0f}")
print(f"FTS Requirements:         ${fts_requirements:>20,.0f}")
print(f"FTS Actual Funding:       ${fts_total:>20,.0f}")

print("\n" + "=" * 60)
print("KEY DIFFERENCES")
print("=" * 60)
print("""
1. WHAT FTS PROVIDES THAT WE DON'T HAVE:
   - 'requirements': Total funding REQUESTED for humanitarian response
   - 'funding': Actual funding RECEIVED (from all sources)
   - 'percentFunded': Coverage rate (funding / requirements)
   - This shows the FUNDING GAP - unmet humanitarian needs

2. WHAT WE CURRENTLY HAVE:
   - CERF: UN emergency fund allocations (subset of total funding)
   - CBPF: Country pooled fund budgets (subset of total funding)
   - HRP: Humanitarian Response Plan requirements (similar to FTS requirements)

3. WHY FTS IS VALUABLE:
   - Tracks ACTUAL money received vs requested
   - Shows funding efficiency (% funded)
   - Covers ALL humanitarian funding, not just UN pooled funds
   - Better for predicting funding gaps and needs

4. RECOMMENDATION:
   - YES, include FTS data
   - Key features to add:
     * FTS_Requirements (total asked)
     * FTS_Funding (total received)
     * FTS_Percent_Funded (coverage rate)
     * FTS_Funding_Gap (requirements - funding)
""")

# Check for a sample country comparison
print("\n--- Sample Country Comparison (Afghanistan 2024) ---")
afg_fts = fts_recent[(fts_recent['ISO3'] == 'AFG') & (fts_recent['year'] == 2024)]
afg_our = merged[(merged['ISO3'] == 'AFG') & (merged['Year'] == 2024)]

if len(afg_fts) > 0:
    print(f"FTS Requirements: ${afg_fts['requirements'].values[0]:,.0f}")
    print(f"FTS Funding: ${afg_fts['funding'].values[0]:,.0f}")
    print(f"FTS % Funded: {afg_fts['percentFunded'].values[0]:.1f}%")
if len(afg_our) > 0:
    print(f"Our Total_Funding: ${afg_our['Total_Funding'].values[0]:,.0f}")
    print(f"Our HRP_Requirements: ${afg_our['HRP_Revised_Requirements'].values[0]:,.0f}")
