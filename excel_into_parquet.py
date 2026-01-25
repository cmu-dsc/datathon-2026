import pandas as pd
import os
import re
from pathlib import Path

def extract_month_year(filename):
    """
    Extract month and year from filename in MONTH_YEAR format.
    Handles various filename patterns like:
    - 20221205_inform_severity_-_november_2022.xlsx -> november_2022
    - 202501_INFORM_Severity_-_January_2025.xlsx -> January_2025
    - 202511_inform_severity_-_late_november_2025_.xlsx -> november_2025
    """
    # Remove file extension
    name = filename.replace('.xlsx', '').replace('.XLSX', '')
    
    # Try to find pattern like "month_year" or "month_year_" at the end
    # Look for patterns like: word(s)_YYYY or YYYY_word(s)
    # Common pattern: something_month_year or something_YYYYMM
    
    # Pattern 1: Look for "month_year" pattern (e.g., "november_2022", "January_2025")
    # This matches: word(s) followed by underscore followed by 4-digit year
    pattern1 = r'([a-zA-Z]+(?:_[a-zA-Z]+)*)_(\d{4})(?:_|$)'
    match = re.search(pattern1, name, re.IGNORECASE)
    if match:
        month_part = match.group(1)
        year = match.group(2)
        # Convert month to lowercase for consistency, but keep original format
        month = month_part.lower()
        return f"{month}_{year}"
    
    # Pattern 2: If filename starts with YYYYMM, extract from that
    # e.g., "202501_INFORM_Severity_-_January_2025" -> "January_2025"
    pattern2 = r'(\d{6})_.*?([a-zA-Z]+(?:_[a-zA-Z]+)*)_(\d{4})'
    match = re.search(pattern2, name, re.IGNORECASE)
    if match:
        month_part = match.group(2)
        year = match.group(3)
        month = month_part.lower()
        return f"{month}_{year}"
    
    # Fallback: try to extract from any month_year pattern
    pattern3 = r'([a-zA-Z]+)_(\d{4})'
    matches = re.findall(pattern3, name, re.IGNORECASE)
    if matches:
        # Take the last match (most likely to be the month_year we want)
        month, year = matches[-1]
        return f"{month.lower()}_{year}"
    
    # If no pattern matches, return a placeholder
    return "unknown_date"

def process_excel_files():
    """
    Process all Excel files in inform_data directory and combine into one CSV and parquet file.
    """
    # Get the directory path
    inform_data_dir = Path(__file__).parent / "inform_data"
    
    # Get all Excel files
    excel_files = list(inform_data_dir.glob("*.xlsx")) + list(inform_data_dir.glob("*.XLSX"))
    
    if not excel_files:
        print("No Excel files found in inform_data directory")
        return
    
    print(f"Found {len(excel_files)} Excel files to process")
    print("TESTING: Processing only the first file...")
    
    dataframes = []
    
    # TEST: Only process first file
    excel_files = excel_files[:1]
    
    for excel_file in excel_files:
        try:
            print(f"Processing: {excel_file.name}")
            
            # Extract month_year from filename
            month_year = extract_month_year(excel_file.name)
            print(f"  Extracted date: {month_year}")
            
            # Read the specific sheet
            # header=1 means use row 1 (second row, 0-indexed) as column names
            # This automatically skips row 0 (first row)
            df = pd.read_excel(excel_file, sheet_name="INFORM Severity - country", header=1)
            
            # Drop the first data row (which is row 2 in the original Excel file, the third row)
            # After using header=1, row 2 becomes index 0 in the dataframe
            if len(df) > 0:
                df = df.drop(df.index[0]).reset_index(drop=True)
            
            # Add month_year column
            df['month_year'] = month_year
            
            dataframes.append(df)
            print(f"  Successfully loaded {len(df)} rows")
            
        except Exception as e:
            print(f"  Error processing {excel_file.name}: {str(e)}")
            continue
    
    if not dataframes:
        print("No dataframes were successfully loaded")
        return
    
    # Concatenate all dataframes
    print("\nConcatenating all dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows in combined dataframe: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    # Print all rows from combined dataframe
    print("\nAll rows in combined dataframe:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(combined_df.to_string())
    
    # Save to CSV file
    csv_path = Path(__file__).parent / "inform_severity_combined.csv"
    print(f"\nSaving CSV to: {csv_path}")
    combined_df.to_csv(csv_path, index=False)
    print(f"Successfully saved CSV file with {len(combined_df)} rows")
    
    # Save to parquet file
    parquet_path = Path(__file__).parent / "inform_severity_combined.parquet"
    print(f"\nSaving parquet to: {parquet_path}")
    combined_df.to_parquet(parquet_path, index=False)
    print(f"Successfully saved parquet file with {len(combined_df)} rows")
    
    return csv_path, parquet_path

if __name__ == "__main__":
    output_paths = process_excel_files()
    if output_paths:
        csv_path, parquet_path = output_paths
        print(f"\n✓ Complete! Files saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Parquet: {parquet_path}")
