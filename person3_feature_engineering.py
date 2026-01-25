"""
Person 3: Predictor Variables + Feature Engineering
Datathon 2026 - Feature Engineering Pipeline

This script handles:
- Task 3.1: Download World Bank Data (GDP, Population, Inflation)
- Task 3.2-3.3: Process Population Data & Calculate Demographics
- Task 3.4: Process HPC Cluster Data
- Task 3.5: Create Master Predictor Dataset
- Task 3.6: Handle Missing Data
- Task 3.7: Create Derived Features
- Task 3.8-3.9: Crisis Type Categorization & Regional Classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK 3.1: World Bank Data
# ============================================================================

def download_world_bank_data():
    """Download World Bank indicators using wbdata package."""
    print("=" * 60)
    print("TASK 3.1: Downloading World Bank Data")
    print("=" * 60)
    
    try:
        import wbdata
        from datetime import datetime
        
        # Define date range
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        print("Downloading GDP per capita...")
        gdp = wbdata.get_dataframe(
            {"NY.GDP.PCAP.KD": "GDP_Per_Capita"},
            date=(start_date, end_date)
        )
        gdp = gdp.reset_index()
        gdp.columns = ['Country_Name', 'Year', 'GDP_Per_Capita']
        gdp['Year'] = pd.to_datetime(gdp['Year']).dt.year
        print(f"  GDP data: {len(gdp)} rows")
        
        print("Downloading Total Population...")
        pop = wbdata.get_dataframe(
            {"SP.POP.TOTL": "WB_Population"},
            date=(start_date, end_date)
        )
        pop = pop.reset_index()
        pop.columns = ['Country_Name', 'Year', 'WB_Population']
        pop['Year'] = pd.to_datetime(pop['Year']).dt.year
        print(f"  Population data: {len(pop)} rows")
        
        print("Downloading Inflation Rate...")
        inflation = wbdata.get_dataframe(
            {"FP.CPI.TOTL.ZG": "Inflation_Rate"},
            date=(start_date, end_date)
        )
        inflation = inflation.reset_index()
        inflation.columns = ['Country_Name', 'Year', 'Inflation_Rate']
        inflation['Year'] = pd.to_datetime(inflation['Year']).dt.year
        print(f"  Inflation data: {len(inflation)} rows")
        
        # Merge all World Bank data
        wb_data = gdp.merge(pop, on=['Country_Name', 'Year'], how='outer')
        wb_data = wb_data.merge(inflation, on=['Country_Name', 'Year'], how='outer')
        
        # Create country name to ISO3 mapping using wbdata
        print("Creating country code mapping...")
        countries = wbdata.get_countries()
        name_to_iso3 = {c['name']: c['id'] for c in countries}
        wb_data['ISO3'] = wb_data['Country_Name'].map(name_to_iso3)
        
        print(f"World Bank data compiled: {len(wb_data)} rows")
        return wb_data
        
    except Exception as e:
        print(f"Error downloading World Bank data: {e}")
        print("Creating placeholder World Bank data from existing sources...")
        return None


def create_fallback_world_bank_data(merged_df):
    """Create fallback economic indicators if World Bank API fails."""
    print("Creating fallback economic indicators...")
    
    # Use existing data to create placeholders
    # We'll use regional averages based on known data
    
    # Create a basic structure
    wb_data = merged_df[['ISO3', 'year_num', 'Country', 'Region']].copy()
    wb_data = wb_data.rename(columns={'year_num': 'Year'})
    
    # Regional GDP estimates (rough approximations for humanitarian contexts)
    regional_gdp = {
        'Africa': 1500,
        'Middle east': 3000,
        'Asia': 2000,
        'Americas': 5000,
        'Europe': 8000,
        'Oceania': 3000
    }
    
    wb_data['GDP_Per_Capita'] = wb_data['Region'].map(
        lambda x: regional_gdp.get(x, 2000) if pd.notna(x) else 2000
    )
    
    # Add some variation based on severity (lower GDP for higher severity countries)
    wb_data['WB_Population'] = np.nan  # Will be filled from other sources
    wb_data['Inflation_Rate'] = 10.0  # Default moderate inflation
    
    return wb_data


# ============================================================================
# TASK 3.2-3.3: Process Population Data
# ============================================================================

def process_population_data():
    """Process OCHA COD population data."""
    print("\n" + "=" * 60)
    print("TASK 3.2-3.3: Processing Population Data")
    print("=" * 60)
    
    # Load population data (skip header row)
    pop = pd.read_csv('data/geo_mismatch/cod_population_admin0.csv')
    
    # Skip the HXL tag row
    if pop.iloc[0]['ISO3'] == '#country+code':
        pop = pop.iloc[1:].reset_index(drop=True)
    
    # Convert numeric columns
    pop['Population'] = pd.to_numeric(pop['Population'], errors='coerce')
    pop['Reference_year'] = pd.to_numeric(pop['Reference_year'], errors='coerce')
    pop['Age_min'] = pd.to_numeric(pop['Age_min'], errors='coerce')
    pop['Age_max'] = pd.to_numeric(pop['Age_max'], errors='coerce')
    
    print(f"Loaded population data: {len(pop)} rows")
    print(f"Countries: {pop['ISO3'].nunique()}")
    print(f"Years: {pop['Reference_year'].unique()}")
    
    # ----- Total Population by Country-Year -----
    # Filter for total population entries
    total_pop = pop[pop['Population_group'].str.contains('TL|total', case=False, na=False)]
    total_pop_agg = total_pop.groupby(['ISO3', 'Reference_year'])['Population'].sum().reset_index()
    total_pop_agg.columns = ['ISO3', 'Year', 'Total_Population']
    print(f"Total population records: {len(total_pop_agg)}")
    
    # ----- IDP Population -----
    idp_pop = pop[pop['Population_group'].str.contains('IDP|displaced', case=False, na=False)]
    if len(idp_pop) > 0:
        idp_agg = idp_pop.groupby(['ISO3', 'Reference_year'])['Population'].sum().reset_index()
        idp_agg.columns = ['ISO3', 'Year', 'IDP_Population']
        print(f"IDP records: {len(idp_agg)}")
    else:
        idp_agg = pd.DataFrame(columns=['ISO3', 'Year', 'IDP_Population'])
        print("No IDP data found")
    
    # ----- Refugee Population -----
    refugee_pop = pop[pop['Population_group'].str.contains('REF|refugee', case=False, na=False)]
    if len(refugee_pop) > 0:
        refugee_agg = refugee_pop.groupby(['ISO3', 'Reference_year'])['Population'].sum().reset_index()
        refugee_agg.columns = ['ISO3', 'Year', 'Refugee_Population']
        print(f"Refugee records: {len(refugee_agg)}")
    else:
        refugee_agg = pd.DataFrame(columns=['ISO3', 'Year', 'Refugee_Population'])
        print("No refugee data found")
    
    # ----- Vulnerable Population (children <5 and elderly 60+) -----
    vulnerable = pop[
        ((pop['Age_max'] <= 5) | (pop['Age_min'] >= 60)) &
        (pop['Population_group'].str.contains('TL|total', case=False, na=False))
    ]
    if len(vulnerable) > 0:
        vuln_agg = vulnerable.groupby(['ISO3', 'Reference_year'])['Population'].sum().reset_index()
        vuln_agg.columns = ['ISO3', 'Year', 'Vulnerable_Population']
        print(f"Vulnerable population records: {len(vuln_agg)}")
    else:
        vuln_agg = pd.DataFrame(columns=['ISO3', 'Year', 'Vulnerable_Population'])
        print("No age-specific data found for vulnerable calculation")
    
    # ----- Merge all demographics -----
    demographics = total_pop_agg.copy()
    
    if len(idp_agg) > 0:
        demographics = demographics.merge(idp_agg, on=['ISO3', 'Year'], how='left')
    else:
        demographics['IDP_Population'] = 0
        
    if len(refugee_agg) > 0:
        demographics = demographics.merge(refugee_agg, on=['ISO3', 'Year'], how='left')
    else:
        demographics['Refugee_Population'] = 0
        
    if len(vuln_agg) > 0:
        demographics = demographics.merge(vuln_agg, on=['ISO3', 'Year'], how='left')
        demographics['Vulnerable_Pop_Pct'] = (
            demographics['Vulnerable_Population'] / demographics['Total_Population'] * 100
        ).fillna(0)
    else:
        demographics['Vulnerable_Population'] = 0
        demographics['Vulnerable_Pop_Pct'] = 35.0  # Default estimate
    
    # Fill NaN with 0 for displacement data
    demographics['IDP_Population'] = demographics['IDP_Population'].fillna(0)
    demographics['Refugee_Population'] = demographics['Refugee_Population'].fillna(0)
    
    print(f"\nDemographics compiled: {len(demographics)} country-year combinations")
    
    return demographics


# ============================================================================
# TASK 3.4: Process HPC Cluster Data
# ============================================================================

def process_hpc_cluster_data():
    """Process HPC Humanitarian Needs Overview cluster data."""
    print("\n" + "=" * 60)
    print("TASK 3.4: Processing HPC Cluster Data")
    print("=" * 60)
    
    # Load all HPC HNO files
    hpc_files = [
        ('data/geo_mismatch/hpc_hno_2024.csv', 2024),
        ('data/geo_mismatch/hpc_hno_2025.csv', 2025),
        ('data/geo_mismatch/hpc_hno_2026.csv', 2026),
    ]
    
    all_hpc = []
    
    for file_path, year in hpc_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Skip HXL tag row if present
            if df.iloc[0]['Country ISO3'] == '#country+code':
                df = df.iloc[1:].reset_index(drop=True)
            
            df['Year'] = year
            all_hpc.append(df)
            print(f"Loaded {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_hpc:
        print("No HPC data loaded")
        return pd.DataFrame()
    
    hpc = pd.concat(all_hpc, ignore_index=True)
    
    # Rename columns for consistency
    hpc = hpc.rename(columns={'Country ISO3': 'ISO3'})
    
    # Convert numeric columns
    for col in ['Population', 'In Need', 'Targeted', 'Affected', 'Reached']:
        if col in hpc.columns:
            hpc[col] = pd.to_numeric(hpc[col], errors='coerce')
    
    print(f"Total HPC data: {len(hpc)} rows")
    print(f"Countries: {hpc['ISO3'].nunique()}")
    print(f"Clusters: {hpc['Cluster'].nunique()}")
    
    # ----- Aggregate by Country-Year -----
    cluster_agg = hpc.groupby(['ISO3', 'Year']).agg({
        'Cluster': 'nunique',  # Number of clusters
        'In Need': 'sum',
        'Targeted': 'sum',
        'Reached': 'sum',
        'Population': 'first'  # Total population
    }).reset_index()
    
    cluster_agg.columns = ['ISO3', 'Year', 'Number_Clusters', 'Total_In_Need', 
                           'Total_Targeted', 'Total_Reached', 'HPC_Population']
    
    # Calculate unmet need
    cluster_agg['Unmet_Need'] = cluster_agg['Total_In_Need'] - cluster_agg['Total_Reached']
    cluster_agg['Coverage_Rate'] = (
        cluster_agg['Total_Reached'] / cluster_agg['Total_In_Need'].replace(0, np.nan) * 100
    ).fillna(0)
    
    print(f"Cluster metrics compiled: {len(cluster_agg)} country-year combinations")
    
    # ----- Identify dominant cluster (for crisis categorization) -----
    dominant_cluster = hpc.groupby(['ISO3', 'Year']).apply(
        lambda x: x.loc[x['In Need'].idxmax(), 'Cluster'] if len(x) > 0 and x['In Need'].notna().any() else 'Unknown'
    ).reset_index()
    dominant_cluster.columns = ['ISO3', 'Year', 'Dominant_Cluster']
    
    cluster_agg = cluster_agg.merge(dominant_cluster, on=['ISO3', 'Year'], how='left')
    
    return cluster_agg


# ============================================================================
# TASK 3.8-3.9: Crisis Type & Regional Classification
# ============================================================================

def categorize_crisis_type(row):
    """Categorize crisis based on dominant cluster and other indicators."""
    cluster = str(row.get('Dominant_Cluster', '')).lower()
    drivers = str(row.get('CERF_Emergency_Types', '')).lower()
    
    # Check CERF emergency types first (more reliable)
    if 'conflict' in drivers or 'violence' in drivers:
        return 'Conflict'
    elif 'drought' in drivers:
        return 'Drought'
    elif 'flood' in drivers:
        return 'Flood'
    elif 'earthquake' in drivers:
        return 'Earthquake'
    elif 'storm' in drivers or 'cyclone' in drivers or 'typhoon' in drivers:
        return 'Storm'
    elif 'epidemic' in drivers or 'disease' in drivers:
        return 'Epidemic'
    
    # Fall back to dominant cluster
    if 'protection' in cluster:
        return 'Conflict'
    elif 'food' in cluster or 'nutrition' in cluster:
        return 'Food Crisis'
    elif 'health' in cluster:
        return 'Health Crisis'
    elif 'shelter' in cluster:
        return 'Displacement'
    elif 'wash' in cluster or 'water' in cluster:
        return 'WASH Crisis'
    else:
        return 'Complex Emergency'


def add_regional_classification(df):
    """Add UN regional classification."""
    
    # UN regional groupings
    region_mapping = {
        # Sub-Saharan Africa
        'AFG': 'Asia', 'AGO': 'Sub-Saharan Africa', 'BDI': 'Sub-Saharan Africa',
        'BEN': 'Sub-Saharan Africa', 'BFA': 'Sub-Saharan Africa', 'BWA': 'Sub-Saharan Africa',
        'CAF': 'Sub-Saharan Africa', 'CIV': 'Sub-Saharan Africa', 'CMR': 'Sub-Saharan Africa',
        'COD': 'Sub-Saharan Africa', 'COG': 'Sub-Saharan Africa', 'COM': 'Sub-Saharan Africa',
        'CPV': 'Sub-Saharan Africa', 'DJI': 'Sub-Saharan Africa', 'ERI': 'Sub-Saharan Africa',
        'ETH': 'Sub-Saharan Africa', 'GAB': 'Sub-Saharan Africa', 'GHA': 'Sub-Saharan Africa',
        'GIN': 'Sub-Saharan Africa', 'GMB': 'Sub-Saharan Africa', 'GNB': 'Sub-Saharan Africa',
        'GNQ': 'Sub-Saharan Africa', 'KEN': 'Sub-Saharan Africa', 'LBR': 'Sub-Saharan Africa',
        'LSO': 'Sub-Saharan Africa', 'MDG': 'Sub-Saharan Africa', 'MLI': 'Sub-Saharan Africa',
        'MOZ': 'Sub-Saharan Africa', 'MRT': 'Sub-Saharan Africa', 'MWI': 'Sub-Saharan Africa',
        'NAM': 'Sub-Saharan Africa', 'NER': 'Sub-Saharan Africa', 'NGA': 'Sub-Saharan Africa',
        'RWA': 'Sub-Saharan Africa', 'SEN': 'Sub-Saharan Africa', 'SLE': 'Sub-Saharan Africa',
        'SOM': 'Sub-Saharan Africa', 'SSD': 'Sub-Saharan Africa', 'SDN': 'Sub-Saharan Africa',
        'SWZ': 'Sub-Saharan Africa', 'TCD': 'Sub-Saharan Africa', 'TGO': 'Sub-Saharan Africa',
        'TZA': 'Sub-Saharan Africa', 'UGA': 'Sub-Saharan Africa', 'ZAF': 'Sub-Saharan Africa',
        'ZMB': 'Sub-Saharan Africa', 'ZWE': 'Sub-Saharan Africa',
        
        # Middle East & North Africa
        'DZA': 'MENA', 'BHR': 'MENA', 'EGY': 'MENA', 'IRN': 'MENA', 'IRQ': 'MENA',
        'ISR': 'MENA', 'JOR': 'MENA', 'KWT': 'MENA', 'LBN': 'MENA', 'LBY': 'MENA',
        'MAR': 'MENA', 'OMN': 'MENA', 'PSE': 'MENA', 'QAT': 'MENA', 'SAU': 'MENA',
        'SYR': 'MENA', 'TUN': 'MENA', 'ARE': 'MENA', 'YEM': 'MENA',
        
        # Asia & Pacific
        'BGD': 'Asia', 'BTN': 'Asia', 'BRN': 'Asia', 'KHM': 'Asia', 'CHN': 'Asia',
        'FJI': 'Asia', 'IND': 'Asia', 'IDN': 'Asia', 'JPN': 'Asia', 'KAZ': 'Asia',
        'KGZ': 'Asia', 'LAO': 'Asia', 'MYS': 'Asia', 'MDV': 'Asia', 'MNG': 'Asia',
        'MMR': 'Asia', 'NPL': 'Asia', 'PRK': 'Asia', 'PAK': 'Asia', 'PNG': 'Asia',
        'PHL': 'Asia', 'KOR': 'Asia', 'LKA': 'Asia', 'TJK': 'Asia', 'THA': 'Asia',
        'TLS': 'Asia', 'TKM': 'Asia', 'UZB': 'Asia', 'VNM': 'Asia', 'VUT': 'Asia',
        
        # Latin America & Caribbean
        'ARG': 'LAC', 'BLZ': 'LAC', 'BOL': 'LAC', 'BRA': 'LAC', 'CHL': 'LAC',
        'COL': 'LAC', 'CRI': 'LAC', 'CUB': 'LAC', 'DOM': 'LAC', 'ECU': 'LAC',
        'SLV': 'LAC', 'GTM': 'LAC', 'GUY': 'LAC', 'HTI': 'LAC', 'HND': 'LAC',
        'JAM': 'LAC', 'MEX': 'LAC', 'NIC': 'LAC', 'PAN': 'LAC', 'PRY': 'LAC',
        'PER': 'LAC', 'SUR': 'LAC', 'TTO': 'LAC', 'URY': 'LAC', 'VEN': 'LAC',
        
        # Europe & Central Asia
        'ALB': 'Europe', 'ARM': 'Europe', 'AZE': 'Europe', 'BLR': 'Europe',
        'BIH': 'Europe', 'BGR': 'Europe', 'HRV': 'Europe', 'CYP': 'Europe',
        'CZE': 'Europe', 'EST': 'Europe', 'GEO': 'Europe', 'HUN': 'Europe',
        'KOS': 'Europe', 'LVA': 'Europe', 'LTU': 'Europe', 'MKD': 'Europe',
        'MDA': 'Europe', 'MNE': 'Europe', 'POL': 'Europe', 'ROU': 'Europe',
        'RUS': 'Europe', 'SRB': 'Europe', 'SVK': 'Europe', 'SVN': 'Europe',
        'TUR': 'Europe', 'UKR': 'Europe',
    }
    
    df['UN_Region'] = df['ISO3'].map(region_mapping).fillna('Other')
    return df


# ============================================================================
# TASK 3.5-3.7: Create Master Dataset & Derived Features
# ============================================================================

def create_predictor_dataset():
    """Create the master predictor dataset with all features."""
    print("\n" + "=" * 60)
    print("TASK 3.5: Creating Master Predictor Dataset")
    print("=" * 60)
    
    # Load Person 1's merged dataset
    base = pd.read_csv('person1_merged_dataset.csv')
    print(f"Loaded base dataset: {len(base)} rows")
    
    # Rename year column for consistency
    base = base.rename(columns={'year_num': 'Year'})
    
    # Get World Bank data
    wb_data = download_world_bank_data()
    
    if wb_data is None or len(wb_data) == 0:
        wb_data = create_fallback_world_bank_data(base)
    
    # Get population demographics
    demographics = process_population_data()
    
    # Get HPC cluster metrics
    cluster_data = process_hpc_cluster_data()
    
    # ----- Merge all datasets -----
    print("\n" + "=" * 60)
    print("Merging all predictor datasets...")
    print("=" * 60)
    
    # Merge World Bank data
    if 'ISO3' in wb_data.columns:
        full_data = base.merge(
            wb_data[['ISO3', 'Year', 'GDP_Per_Capita', 'WB_Population', 'Inflation_Rate']].drop_duplicates(),
            on=['ISO3', 'Year'],
            how='left'
        )
        print(f"After World Bank merge: {len(full_data)} rows")
    else:
        full_data = base.copy()
        full_data['GDP_Per_Capita'] = np.nan
        full_data['WB_Population'] = np.nan
        full_data['Inflation_Rate'] = np.nan
    
    # Merge demographics
    if len(demographics) > 0:
        full_data = full_data.merge(
            demographics[['ISO3', 'Year', 'Total_Population', 'IDP_Population', 
                         'Refugee_Population', 'Vulnerable_Pop_Pct']].drop_duplicates(),
            on=['ISO3', 'Year'],
            how='left'
        )
        print(f"After demographics merge: {len(full_data)} rows")
    
    # Merge cluster data
    if len(cluster_data) > 0:
        full_data = full_data.merge(
            cluster_data[['ISO3', 'Year', 'Number_Clusters', 'Total_In_Need', 
                         'Total_Reached', 'Coverage_Rate', 'Dominant_Cluster']].drop_duplicates(),
            on=['ISO3', 'Year'],
            how='left'
        )
        print(f"After cluster merge: {len(full_data)} rows")
    
    # ----- Handle Missing Data (Task 3.6) -----
    print("\n" + "=" * 60)
    print("TASK 3.6: Handling Missing Data")
    print("=" * 60)
    
    # Fill GDP with regional median
    if 'Region' in full_data.columns:
        full_data['GDP_Per_Capita'] = full_data.groupby('Region')['GDP_Per_Capita'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Fill remaining GDP NaN with global median
    gdp_median = full_data['GDP_Per_Capita'].median()
    full_data['GDP_Per_Capita'] = full_data['GDP_Per_Capita'].fillna(gdp_median if pd.notna(gdp_median) else 2000)
    
    # Fill population using existing data
    if 'Total_Population' in full_data.columns and 'WB_Population' in full_data.columns:
        full_data['Population'] = full_data['Total_Population'].fillna(full_data['WB_Population'])
    elif 'Total_Population' in full_data.columns:
        full_data['Population'] = full_data['Total_Population']
    else:
        full_data['Population'] = full_data.get('WB_Population', 1000000)
    
    # Fill other NaN values
    fill_values = {
        'Inflation_Rate': 10.0,
        'IDP_Population': 0,
        'Refugee_Population': 0,
        'Vulnerable_Pop_Pct': 35.0,
        'Number_Clusters': 5,
        'Total_In_Need': 0,
        'Total_Reached': 0,
        'Coverage_Rate': 0,
    }
    
    for col, default_val in fill_values.items():
        if col in full_data.columns:
            full_data[col] = full_data[col].fillna(default_val)
    
    missing_report = full_data.isnull().sum()
    print("Missing values after imputation:")
    print(missing_report[missing_report > 0].to_string())
    
    # ----- Create Derived Features (Task 3.7) -----
    print("\n" + "=" * 60)
    print("TASK 3.7: Creating Derived Features")
    print("=" * 60)
    
    # People in Need per Capita
    full_data['Need_Per_Capita'] = (
        full_data['Total_In_Need'] / full_data['Population'].replace(0, np.nan)
    ).fillna(0)
    
    # IDP Rate
    full_data['IDP_Rate'] = (
        full_data['IDP_Population'] / full_data['Population'].replace(0, np.nan) * 100
    ).fillna(0)
    
    # Economic Stress Score (low GDP + high inflation = high stress)
    full_data['Economic_Stress'] = (
        (1 / (full_data['GDP_Per_Capita'].replace(0, 1) + 1)) * 
        full_data['Inflation_Rate'].clip(lower=0)
    ).fillna(0)
    
    # Funding per person in need
    full_data['Funding_Per_Person_In_Need'] = (
        full_data['Total_Funding'] / full_data['Total_In_Need'].replace(0, np.nan)
    ).fillna(0)
    
    # Severity-adjusted funding ratio
    full_data['Severity_Funding_Ratio'] = (
        full_data['Total_Funding'] / 
        (full_data['INFORM_Mean'].replace(0, 1) * full_data['Population'].replace(0, 1))
    ).fillna(0)
    
    print("Derived features created:")
    print("  - Need_Per_Capita")
    print("  - IDP_Rate")
    print("  - Economic_Stress")
    print("  - Funding_Per_Person_In_Need")
    print("  - Severity_Funding_Ratio")
    
    # ----- Add Crisis Type Categorization (Task 3.8) -----
    print("\n" + "=" * 60)
    print("TASK 3.8: Categorizing Crisis Types")
    print("=" * 60)
    
    full_data['Crisis_Type'] = full_data.apply(categorize_crisis_type, axis=1)
    print("Crisis type distribution:")
    print(full_data['Crisis_Type'].value_counts().to_string())
    
    # ----- Add Regional Classification (Task 3.9) -----
    print("\n" + "=" * 60)
    print("TASK 3.9: Adding Regional Classification")
    print("=" * 60)
    
    full_data = add_regional_classification(full_data)
    print("Regional distribution:")
    print(full_data['UN_Region'].value_counts().to_string())
    
    return full_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("DATATHON 2026 - PERSON 3 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Create the full predictor dataset
    full_data = create_predictor_dataset()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    print(f"Total rows: {len(full_data)}")
    print(f"Total columns: {len(full_data.columns)}")
    print(f"Countries: {full_data['ISO3'].nunique()}")
    print(f"Years: {full_data['Year'].min():.0f} to {full_data['Year'].max():.0f}")
    
    print("\nColumns:")
    for col in full_data.columns:
        print(f"  - {col}: {full_data[col].dtype}")
    
    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Save predictor dataset
    full_data.to_csv('person3_predictor_variables.csv', index=False)
    print("Saved: person3_predictor_variables.csv")
    
    # Create and save data dictionary
    data_dict = []
    for col in full_data.columns:
        data_dict.append({
            'Variable': col,
            'Type': str(full_data[col].dtype),
            'Non_Null': full_data[col].notna().sum(),
            'Missing_Pct': f"{full_data[col].isna().sum() / len(full_data) * 100:.1f}%",
            'Unique': full_data[col].nunique(),
            'Sample_Values': str(full_data[col].dropna().head(3).tolist())[:50]
        })
    
    pd.DataFrame(data_dict).to_csv('person3_data_dictionary.csv', index=False)
    print("Saved: person3_data_dictionary.csv")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return full_data


if __name__ == "__main__":
    full_data = main()
