#!/usr/bin/env python3
"""
Script to extract total latency and minimum s values from JSON data
and save to an Excel spreadsheet, including tail latency (P95 / T95).
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path,required=True, help="Path to the input directory")
    return parser.parse_args()

def extract_latency_data(json_path):
    """
    Extract total latency and minimum s value from each scene in the JSON file.
    
    Args:
        json_path: Path to the JSON file containing latency data
        
    Returns:
        DataFrame with scene_id, total_latency, and min_s columns
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data for each scene
    results = []
    for scene_id, scene_data in data.items():
        total_latency = scene_data['total_latency']
        s_array = scene_data['s']

        # Option 1: Minimum including boundary 50s
        min_s_all = min(s_array)

        # Option 2: Minimum excluding boundary 50s (more meaningful)
        min_s_interior = min(s_array[1:-1]) if len(s_array) > 2 else min(s_array)

        # Interior stats
        s_interior = s_array[1:-1] if len(s_array) > 2 else s_array
        mean_s = float(np.mean(s_interior))
        median_s = float(np.median(s_interior))
        std_s = float(np.std(s_interior))

        results.append({
            'scene_id': scene_id,
            'total_latency': total_latency,
            'mvsplat_latency': scene_data['mvsplat_latency'],
            'seva_latency': scene_data['seva_latency'],
            'min_s_all': min_s_all,
            'min_s_interior': min_s_interior,
            'mean_s_interior': mean_s,
            'median_s_interior': median_s,
            'std_s_interior': std_s,
            's_array': str(s_array)  # Store full array as string for reference
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('scene_id')
    return df

def save_to_excel(df, output_path):
    """
    Save DataFrame to Excel with multiple sheets for different analyses.
    
    Args:
        df: DataFrame containing the extracted data
        output_path: Path for the output Excel file
    """
    # Precompute tail latencies (P95/T95)
    p95_total = float(df['total_latency'].quantile(0.95))
    p95_mvsplat = float(df['mvsplat_latency'].quantile(0.95))
    p95_seva = float(df['seva_latency'].quantile(0.95))

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All data
        df.to_excel(writer, sheet_name='All_Data', index=False)
        
        # Sheet 2: Summary statistics (now includes T95/P95)
        summary_stats = {
            'Metric': [
                'Total Scenes',
                'Avg Total Latency',
                'Min Total Latency',
                'Max Total Latency',
                'Std Total Latency',
                'P95 Total Latency (T95)',
                'Avg MVSplat Latency',
                'P95 MVSplat Latency',
                'Avg SEVA Latency',
                'P95 SEVA Latency',
                'Avg Min S (interior)',
                'Min of Min S (interior)',
                'Max of Min S (interior)',
                'Scenes with Min S = 0'
            ],
            'Value': [
                len(df),
                df['total_latency'].mean(),
                df['total_latency'].min(),
                df['total_latency'].max(),
                df['total_latency'].std(),
                p95_total,
                df['mvsplat_latency'].mean(),
                p95_mvsplat,
                df['seva_latency'].mean(),
                p95_seva,
                df['min_s_interior'].mean(),
                df['min_s_interior'].min(),
                df['min_s_interior'].max(),
                int((df['min_s_interior'] == 0).sum())
            ]
        }
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 3: Sorted by total latency (ascending)
        sorted_by_latency = df[['scene_id', 'total_latency', 'min_s_interior']].copy()
        sorted_by_latency = sorted_by_latency.sort_values('total_latency')
        sorted_by_latency.to_excel(writer, sheet_name='Sorted_by_Latency', index=False)
        
        # Sheet 4: Sorted by min_s (ascending)
        sorted_by_min_s = df[['scene_id', 'total_latency', 'min_s_interior']].copy()
        sorted_by_min_s = sorted_by_min_s.sort_values('min_s_interior')
        sorted_by_min_s.to_excel(writer, sheet_name='Sorted_by_Min_S', index=False)
        
        # Sheet 5: Correlation analysis
        correlation_data = df[['total_latency', 'mvsplat_latency', 'seva_latency', 
                               'min_s_interior', 'mean_s_interior', 'median_s_interior']]
        correlation_matrix = correlation_data.corr()
        correlation_matrix.to_excel(writer, sheet_name='Correlations')
        
        # Optional: Sheet 6 with a small percentile table for quick reference
        percentiles = [0.5, 0.9, 0.95, 0.99]
        perc_table = pd.DataFrame({
            'Percentile': ['P50', 'P90', 'P95', 'P99'],
            'Total Latency': [float(df['total_latency'].quantile(p)) for p in percentiles],
            'MVSplat Latency': [float(df['mvsplat_latency'].quantile(p)) for p in percentiles],
            'SEVA Latency': [float(df['seva_latency'].quantile(p)) for p in percentiles],
        })
        perc_table.to_excel(writer, sheet_name='Latency_Percentiles', index=False)
        
    print(f"Excel file saved to: {output_path}")

def main():
    """Main function to run the extraction and save to Excel."""
    args = get_args()
    print("Directory:", args.dir)
    dirc = args.dir
    json_path = dirc / "latency.json"
    # File paths
    output_path = Path(f"latency_analysis_{dirc.name}.xlsx")
    
    # Check if JSON file exists
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    
    # Extract data
    print(f"Loading data from {json_path}...")
    df = extract_latency_data(json_path)
    
    # Compute tail latencies
    p95_total = float(df['total_latency'].quantile(0.95))
    p95_mvsplat = float(df['mvsplat_latency'].quantile(0.95))
    p95_seva = float(df['seva_latency'].quantile(0.95))
    avg_mvsplat = float(df['mvsplat_latency'].mean())
    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total scenes: {len(df)}")
    print(f"Average total latency: {df['total_latency'].mean():.2f} seconds")
    print(f"Average MVSplat latency: {avg_mvsplat:.2f} seconds")
    print(f"P95 total latency: {p95_total:.2f} seconds")
    print(f"P95 mvsplat latency: {p95_mvsplat:.2f} seconds")
    print(f"P95 SEVA latency: {p95_seva:.2f} seconds")
    print(f"Average min S value (excluding boundaries): {df['min_s_interior'].mean():.2f}")
    # print(f"Number of scenes with min S = 0: {int((df['min_s_interior'] == 0).sum())}")
    
    # Save to Excel
    save_to_excel(df, output_path)
    
    # Also save as CSV for compatibility
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to: {csv_path}")
    

if __name__ == "__main__":
    main()
