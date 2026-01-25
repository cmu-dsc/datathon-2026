"""
Person 1: Data Infrastructure + Model Building
Datathon 2026 - Data Merging Pipeline

This script handles:
- Task 1.1: Load and standardize INFORM data
- Task 1.2: Load and standardize HRP/Financial data  
- Task 1.3: Merge INFORM + Financial data
- Task 1.4: Create country mapping
- Task 1.5: Data quality report
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK 1.1: Load and Standardize INFORM Data
# ============================================================================

def load_inform_data():
    """Load and clean INFORM severity data."""
    print("=" * 60)
    print("TASK 1.1: Loading INFORM Severity Data")
    print("=" * 60)
    
    inform = pd.read_csv('inform_severity_combined.csv')
    print(f"Loaded {len(inform)} rows, {len(inform.columns)} columns")
    
    # Clean month names
    month_mapping = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'late_november': 11, 'inform_severity_mid_december': 12,
        'unknown_date': None
    }
    
    inform['month_num'] = inform['month'].map(month_mapping)
    
    # Convert year to numeric
    inform['year_num'] = pd.to_numeric(inform['year'], errors='coerce')
    
    # Create Date column
    def create_date(row):
        if pd.notna(row['year_num']) and pd.notna(row['month_num']):
            try:
                return pd.Timestamp(year=int(row['year_num']), month=int(row['month_num']), day=1)
            except:
                return pd.NaT
        return pd.NaT
    
    inform['Date'] = inform.apply(create_date, axis=1)
    
    # Sort by country and date
    inform = inform.sort_values(['ISO3', 'Date']).reset_index(drop=True)
    
    # Flag missing months
    inform['has_valid_date'] = inform['Date'].notna()
    
    print(f"Valid dates: {inform['has_valid_date'].sum()} / {len(inform)}")
    print(f"Date range: {inform['Date'].min()} to {inform['Date'].max()}")
    print(f"Countries: {inform['ISO3'].nunique()}")
    print(f"Crises: {inform['CRISIS ID'].nunique()}")
    
    return inform


# ============================================================================
# TASK 1.2: Load and Standardize Financial Data
# ============================================================================

def load_hrp_data():
    """Load and clean Humanitarian Response Plans data."""
    print("\n" + "=" * 60)
    print("TASK 1.2a: Loading HRP Data")
    print("=" * 60)
    
    hrp = pd.read_csv('data/geo_mismatch/humanitarian-response-plans.csv')
    
    # Skip header row if present
    if hrp.iloc[0]['code'] == '#response+code':
        hrp = hrp.iloc[1:].reset_index(drop=True)
    
    print(f"Loaded {len(hrp)} HRP plans")
    
    # Parse dates
    hrp['StartDate'] = pd.to_datetime(hrp['startDate'], errors='coerce')
    hrp['EndDate'] = pd.to_datetime(hrp['endDate'], errors='coerce')
    
    # Calculate duration
    hrp['Duration_Months'] = ((hrp['EndDate'] - hrp['StartDate']).dt.days / 30).round()
    
    # Extract year
    hrp['Start_Year'] = hrp['StartDate'].dt.year
    
    # Handle multi-country plans (explode comma-separated locations)
    hrp['Location'] = hrp['locations'].str.split(',')
    hrp = hrp.explode('Location')
    hrp['Location'] = hrp['Location'].str.strip()
    
    # Convert funding to numeric
    hrp['origRequirements'] = pd.to_numeric(hrp['origRequirements'], errors='coerce')
    hrp['revisedRequirements'] = pd.to_numeric(hrp['revisedRequirements'], errors='coerce')
    
    print(f"After exploding multi-country plans: {len(hrp)} rows")
    print(f"Countries covered: {hrp['Location'].nunique()}")
    print(f"Date range: {hrp['StartDate'].min()} to {hrp['EndDate'].max()}")
    print(f"Total funding requested: ${hrp['revisedRequirements'].sum():,.0f}")
    
    return hrp


def load_cerf_data():
    """Load CERF allocations data."""
    print("\n" + "=" * 60)
    print("TASK 1.2b: Loading CERF Allocations Data")
    print("=" * 60)
    
    cerf = pd.read_csv('data/project_targeting/Data_ CERF Donor Contributions and Allocations - allocations.csv')
    print(f"Loaded {len(cerf)} CERF allocations")
    print(f"Countries: {cerf['countryCode'].nunique()}")
    print(f"Years: {cerf['year'].min()} to {cerf['year'].max()}")
    print(f"Total allocated: ${cerf['totalAmountApproved'].sum():,.0f}")
    
    return cerf


def load_cbpf_data():
    """Load CBPF projects data."""
    print("\n" + "=" * 60)
    print("TASK 1.2c: Loading CBPF Projects Data")
    print("=" * 60)
    
    cbpf = pd.read_csv('data/project_targeting/Data_ Country Based Pooled Funds (CBPF) - Projects.csv')
    print(f"Loaded {len(cbpf)} CBPF projects")
    print(f"Countries (fund names): {cbpf['PooledFundName'].nunique()}")
    print(f"Years: {cbpf['AllocationYear'].min()} to {cbpf['AllocationYear'].max()}")
    print(f"Total budget: ${cbpf['Budget'].sum():,.0f}")
    
    return cbpf


# ============================================================================
# TASK 1.3: Merge INFORM + Financial Data
# ============================================================================

def create_country_mapping(inform, cerf, cbpf):
    """Create a master country code mapping."""
    print("\n" + "=" * 60)
    print("TASK 1.4: Creating Country Mapping")
    print("=" * 60)
    
    # Get unique countries from INFORM
    inform_countries = inform[['ISO3', 'COUNTRY', 'Regions']].drop_duplicates()
    inform_countries = inform_countries.rename(columns={'COUNTRY': 'Country_Name', 'Regions': 'Region'})
    
    # Get CERF country mapping
    cerf_countries = cerf[['countryCode', 'countryName', 'continentName', 'regionName']].drop_duplicates()
    cerf_countries = cerf_countries.rename(columns={
        'countryCode': 'ISO3',
        'countryName': 'CERF_Country_Name',
        'continentName': 'Continent',
        'regionName': 'CERF_Region'
    })
    
    # Merge mappings
    country_mapping = inform_countries.merge(cerf_countries, on='ISO3', how='outer')
    
    # Create CBPF name to ISO3 mapping
    # CBPF uses country names as PooledFundName - we need to map these to ISO3
    cbpf_names = cbpf['PooledFundName'].unique()
    
    # Create mapping from CBPF names to ISO3 using CERF data
    name_to_iso3 = dict(zip(cerf['countryName'], cerf['countryCode']))
    
    # Add common variations
    name_to_iso3.update({
        'Afghanistan': 'AFG', 'Yemen': 'YEM', 'Syria': 'SYR', 'Sudan': 'SDN',
        'South Sudan': 'SSD', 'Somalia': 'SOM', 'Nigeria': 'NGA', 'Myanmar': 'MMR',
        'Ethiopia': 'ETH', 'DRC': 'COD', 'Democratic Republic of Congo': 'COD',
        'Central African Republic': 'CAF', 'Chad': 'TCD', 'Mali': 'MLI',
        'Niger': 'NER', 'Burkina Faso': 'BFA', 'Cameroon': 'CMR', 'Iraq': 'IRQ',
        'Libya': 'LBY', 'Lebanon': 'LBN', 'Pakistan': 'PAK', 'Ukraine': 'UKR',
        'Haiti': 'HTI', 'Colombia': 'COL', 'Venezuela': 'VEN', 'Bangladesh': 'BGD',
        'Mozambique': 'MOZ', 'Zimbabwe': 'ZWE', 'Malawi': 'MWI', 'Madagascar': 'MDG',
        "Côte d'Ivoire": 'CIV', 'occupied Palestinian territory': 'PSE', 'oPt': 'PSE'
    })
    
    print(f"Country mapping created: {len(country_mapping)} countries")
    
    return country_mapping, name_to_iso3


def merge_inform_with_financials(inform, hrp, cerf, cbpf, name_to_iso3):
    """Merge INFORM data with financial data."""
    print("\n" + "=" * 60)
    print("TASK 1.3: Merging INFORM with Financial Data")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Aggregate CERF allocations by country-year
    # -------------------------------------------------------------------------
    cerf_agg = cerf.groupby(['countryCode', 'year']).agg({
        'totalAmountApproved': 'sum',
        'projectID': 'count',
        'emergencyTypeName': lambda x: ', '.join(x.unique())
    }).reset_index()
    
    cerf_agg = cerf_agg.rename(columns={
        'countryCode': 'ISO3',
        'year': 'year_num',
        'totalAmountApproved': 'CERF_Allocation',
        'projectID': 'CERF_Project_Count',
        'emergencyTypeName': 'CERF_Emergency_Types'
    })
    
    print(f"CERF aggregated: {len(cerf_agg)} country-year combinations")
    
    # -------------------------------------------------------------------------
    # Step 2: Aggregate CBPF by country-year (map names to ISO3)
    # -------------------------------------------------------------------------
    cbpf['ISO3'] = cbpf['PooledFundName'].map(name_to_iso3)
    
    cbpf_agg = cbpf.groupby(['ISO3', 'AllocationYear']).agg({
        'Budget': 'sum',
        'ChfId': 'count',
        'Men': 'sum',
        'Women': 'sum',
        'Boys': 'sum',
        'Girls': 'sum'
    }).reset_index()
    
    cbpf_agg = cbpf_agg.rename(columns={
        'AllocationYear': 'year_num',
        'Budget': 'CBPF_Budget',
        'ChfId': 'CBPF_Project_Count'
    })
    
    cbpf_agg['CBPF_Total_Beneficiaries'] = cbpf_agg[['Men', 'Women', 'Boys', 'Girls']].sum(axis=1)
    
    print(f"CBPF aggregated: {len(cbpf_agg)} country-year combinations")
    
    # -------------------------------------------------------------------------
    # Step 3: Aggregate HRP by country-year
    # -------------------------------------------------------------------------
    hrp_agg = hrp.groupby(['Location', 'Start_Year']).agg({
        'origRequirements': 'sum',
        'revisedRequirements': 'sum',
        'code': 'count',
        'Duration_Months': 'mean'
    }).reset_index()
    
    hrp_agg = hrp_agg.rename(columns={
        'Location': 'ISO3',
        'Start_Year': 'year_num',
        'origRequirements': 'HRP_Original_Requirements',
        'revisedRequirements': 'HRP_Revised_Requirements',
        'code': 'HRP_Plan_Count',
        'Duration_Months': 'HRP_Avg_Duration_Months'
    })
    
    print(f"HRP aggregated: {len(hrp_agg)} country-year combinations")
    
    # -------------------------------------------------------------------------
    # Step 4: Merge with INFORM (aggregate INFORM by country-year first)
    # -------------------------------------------------------------------------
    inform_agg = inform.groupby(['ISO3', 'year_num']).agg({
        'INFORM Severity Index': ['mean', 'std', 'min', 'max', 'first', 'last'],
        'CRISIS ID': 'nunique',
        'COUNTRY': 'first',
        'Regions': 'first',
        'People in need': 'mean',
        'Complexity of the crisis': 'mean',
        'Impact of the crisis': 'mean'
    }).reset_index()
    
    # Flatten column names
    inform_agg.columns = ['ISO3', 'year_num', 
                          'INFORM_Mean', 'INFORM_Std', 'INFORM_Min', 'INFORM_Max', 
                          'INFORM_Start', 'INFORM_End',
                          'Crisis_Count', 'Country', 'Region',
                          'People_In_Need_Avg', 'Complexity_Avg', 'Impact_Avg']
    
    # Calculate change metrics
    inform_agg['INFORM_Change'] = inform_agg['INFORM_End'] - inform_agg['INFORM_Start']
    inform_agg['INFORM_Range'] = inform_agg['INFORM_Max'] - inform_agg['INFORM_Min']
    
    print(f"INFORM aggregated: {len(inform_agg)} country-year combinations")
    
    # -------------------------------------------------------------------------
    # Step 5: Merge all datasets
    # -------------------------------------------------------------------------
    merged = inform_agg.merge(cerf_agg, on=['ISO3', 'year_num'], how='left')
    merged = merged.merge(cbpf_agg[['ISO3', 'year_num', 'CBPF_Budget', 'CBPF_Project_Count', 'CBPF_Total_Beneficiaries']], 
                          on=['ISO3', 'year_num'], how='left')
    merged = merged.merge(hrp_agg, on=['ISO3', 'year_num'], how='left')
    
    # Calculate total funding
    merged['Total_Funding'] = merged[['CERF_Allocation', 'CBPF_Budget', 'HRP_Revised_Requirements']].sum(axis=1, skipna=True)
    
    # Fill NaN funding with 0 where we have INFORM data
    funding_cols = ['CERF_Allocation', 'CBPF_Budget', 'HRP_Revised_Requirements', 
                    'CERF_Project_Count', 'CBPF_Project_Count', 'HRP_Plan_Count']
    for col in funding_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    
    print(f"\nFinal merged dataset: {len(merged)} rows")
    print(f"Countries: {merged['ISO3'].nunique()}")
    print(f"Years: {merged['year_num'].min():.0f} to {merged['year_num'].max():.0f}")
    
    # Check coverage
    has_cerf = (merged['CERF_Allocation'] > 0).sum()
    has_cbpf = (merged['CBPF_Budget'] > 0).sum()
    has_hrp = (merged['HRP_Revised_Requirements'] > 0).sum()
    has_any_funding = (merged['Total_Funding'] > 0).sum()
    
    print(f"\nFunding coverage:")
    print(f"  - Has CERF data: {has_cerf} ({100*has_cerf/len(merged):.1f}%)")
    print(f"  - Has CBPF data: {has_cbpf} ({100*has_cbpf/len(merged):.1f}%)")
    print(f"  - Has HRP data: {has_hrp} ({100*has_hrp/len(merged):.1f}%)")
    print(f"  - Has any funding: {has_any_funding} ({100*has_any_funding/len(merged):.1f}%)")
    
    return merged


# ============================================================================
# TASK 1.5: Data Quality Report
# ============================================================================

def generate_quality_report(merged, inform):
    """Generate data quality report."""
    print("\n" + "=" * 60)
    print("TASK 1.5: Data Quality Report")
    print("=" * 60)
    
    report = {
        'Total_Country_Years': len(merged),
        'Countries_Covered': merged['ISO3'].nunique(),
        'Year_Range': f"{merged['year_num'].min():.0f} to {merged['year_num'].max():.0f}",
        'Total_Crises': merged['Crisis_Count'].sum(),
        
        # INFORM coverage
        'Missing_INFORM_Mean': merged['INFORM_Mean'].isna().sum(),
        'Avg_INFORM_Severity': merged['INFORM_Mean'].mean(),
        'Std_INFORM_Severity': merged['INFORM_Mean'].std(),
        
        # Funding coverage
        'Has_CERF_Funding': (merged['CERF_Allocation'] > 0).sum(),
        'Has_CBPF_Funding': (merged['CBPF_Budget'] > 0).sum(),
        'Has_HRP_Funding': (merged['HRP_Revised_Requirements'] > 0).sum(),
        
        # Funding totals
        'Total_CERF_Allocation': merged['CERF_Allocation'].sum(),
        'Total_CBPF_Budget': merged['CBPF_Budget'].sum(),
        'Total_HRP_Requirements': merged['HRP_Revised_Requirements'].sum(),
        
        # Averages
        'Avg_CERF_Per_Country_Year': merged[merged['CERF_Allocation'] > 0]['CERF_Allocation'].mean(),
        'Avg_CBPF_Per_Country_Year': merged[merged['CBPF_Budget'] > 0]['CBPF_Budget'].mean(),
    }
    
    print("\n--- Summary Statistics ---")
    for key, value in report.items():
        if isinstance(value, float):
            if value > 1000000:
                print(f"  {key}: ${value:,.0f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Identify outliers
    print("\n--- Potential Outliers ---")
    
    # High severity with no funding
    high_sev_no_fund = merged[(merged['INFORM_Mean'] >= 4.0) & (merged['Total_Funding'] == 0)]
    print(f"High severity (>=4.0) with no funding: {len(high_sev_no_fund)} country-years")
    if len(high_sev_no_fund) > 0:
        print(f"  Countries: {high_sev_no_fund['ISO3'].unique()[:10].tolist()}")
    
    # Large funding changes
    merged['Funding_Per_Severity'] = merged['Total_Funding'] / merged['INFORM_Mean'].replace(0, np.nan)
    
    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("DATATHON 2026 - PERSON 1 DATA PIPELINE")
    print("=" * 60)
    
    # Task 1.1: Load INFORM data
    inform = load_inform_data()
    
    # Task 1.2: Load financial data
    hrp = load_hrp_data()
    cerf = load_cerf_data()
    cbpf = load_cbpf_data()
    
    # Task 1.4: Create country mapping (needed for merging)
    country_mapping, name_to_iso3 = create_country_mapping(inform, cerf, cbpf)
    
    # Task 1.3: Merge datasets
    merged = merge_inform_with_financials(inform, hrp, cerf, cbpf, name_to_iso3)
    
    # Task 1.5: Quality report
    report = generate_quality_report(merged, inform)
    
    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Save cleaned INFORM data
    inform.to_csv('person1_inform_clean.csv', index=False)
    print("Saved: person1_inform_clean.csv")
    
    # Save merged dataset
    merged.to_csv('person1_merged_dataset.csv', index=False)
    print("Saved: person1_merged_dataset.csv")
    
    # Save country mapping
    country_mapping.to_csv('person1_country_mapping.csv', index=False)
    print("Saved: person1_country_mapping.csv")
    
    # Save quality report
    pd.DataFrame([report]).T.to_csv('person1_data_quality_report.csv')
    print("Saved: person1_data_quality_report.csv")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return merged, inform, country_mapping


if __name__ == "__main__":
    merged, inform, country_mapping = main()
