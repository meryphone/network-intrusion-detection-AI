#!/usr/bin/env python3
"""
Compare Results - Cross-Dataset Evaluation Aggregator.

This script collates all evaluation CSVs across datasets and models
into a unified summary for easy comparison.

Usage:
    python src/evaluation/compare_results.py
    python src/evaluation/compare_results.py --output evaluation/summary.csv
    python src/evaluation/compare_results.py --format markdown

Output:
    - evaluation/summary.csv: Combined metrics across all experiments
    - evaluation/summary_by_model.csv: Best results per model
    - evaluation/summary_by_dataset.csv: Best results per dataset
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVALUATION_DIR = PROJECT_ROOT / 'evaluation'

# Models and their evaluation file patterns
MODELS = {
    'isolation_forest': 'isolation_forest_evaluation.csv',
    'sgd': 'sgd_evaluation.csv',
    'mlp': 'mlp_evaluation.csv',
    'ensemble': 'ensemble_evaluation.csv',
}

# Key metrics to include in summaries (in order of priority)
KEY_METRICS = [
    'accuracy',
    'balanced_accuracy',
    'f1_macro',
    'f1_anomaly',
    'precision_anomaly',
    'recall_anomaly',
    'stage2_call_rate',
    'total_time_sec',
]

# Metrics where higher is better
HIGHER_IS_BETTER = [
    'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_anomaly',
    'precision_anomaly', 'recall_anomaly', 'precision_macro', 'recall_macro'
]


def find_evaluation_files() -> List[Dict[str, Any]]:
    """
    Find all evaluation CSV files in the evaluation directory.
    
    Returns:
        List of dicts with 'model', 'dataset', 'path' keys
    """
    files = []
    
    for model_name, eval_filename in MODELS.items():
        model_dir = EVALUATION_DIR / model_name
        
        if not model_dir.exists():
            continue
        
        # Look for dataset subdirectories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            eval_file = dataset_dir / eval_filename
            if eval_file.exists():
                files.append({
                    'model': model_name,
                    'dataset': dataset_dir.name,
                    'path': eval_file,
                })
    
    return files


def load_evaluation_file(file_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load an evaluation CSV file and add metadata columns.
    
    Args:
        file_info: Dict with 'model', 'dataset', 'path' keys
    
    Returns:
        DataFrame with added model and dataset columns, or None on error
    """
    try:
        df = pd.read_csv(file_info['path'])
        df['model'] = file_info['model']
        df['dataset'] = file_info['dataset']
        df['source_file'] = str(file_info['path'].relative_to(PROJECT_ROOT))
        return df
    except Exception as e:
        print(f"Warning: Failed to load {file_info['path']}: {e}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and ensure consistent schema.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized columns
    """
    # Rename common variations
    rename_map = {
        'elapsed_sec': 'total_time_sec',
        'time_sec': 'total_time_sec',
        'balanced_acc': 'balanced_accuracy',
    }
    
    df = df.rename(columns=rename_map)
    return df


def aggregate_results(output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Aggregate all evaluation results into a single DataFrame.
    
    Args:
        output_dir: Optional directory to save output files
    
    Returns:
        Combined DataFrame with all results
    """
    print("=" * 70)
    print("CROSS-DATASET EVALUATION COMPARISON")
    print("=" * 70)
    
    # Find all evaluation files
    files = find_evaluation_files()
    
    if not files:
        print("No evaluation files found.")
        print(f"Expected location: {EVALUATION_DIR}/<model>/<dataset>/*_evaluation.csv")
        return pd.DataFrame()
    
    print(f"\nFound {len(files)} evaluation file(s):")
    for f in files:
        print(f"  - {f['model']}/{f['dataset']}: {f['path'].name}")
    
    # Load and combine all files
    dfs = []
    for file_info in files:
        df = load_evaluation_file(file_info)
        if df is not None:
            df = normalize_columns(df)
            dfs.append(df)
    
    if not dfs:
        print("\nNo valid data loaded.")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    
    # Reorder columns: model, dataset first, then metrics
    priority_cols = ['model', 'dataset']
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]
    
    print(f"\nTotal rows: {len(combined)}")
    print(f"Datasets: {combined['dataset'].unique().tolist()}")
    print(f"Models: {combined['model'].unique().tolist()}")
    
    return combined


def get_best_results(combined: pd.DataFrame, group_by: str, 
                    metric: str = 'f1_macro') -> pd.DataFrame:
    """
    Get best result for each group based on a metric.
    
    Args:
        combined: Combined results DataFrame
        group_by: Column to group by ('model', 'dataset', or both)
        metric: Metric to optimize
    
    Returns:
        DataFrame with best results per group
    """
    if metric not in combined.columns:
        # Try alternative metrics
        for alt_metric in KEY_METRICS:
            if alt_metric in combined.columns:
                metric = alt_metric
                break
        else:
            return combined.drop_duplicates(subset=[group_by])
    
    # Sort and get best
    ascending = metric not in HIGHER_IS_BETTER
    sorted_df = combined.sort_values(metric, ascending=ascending)
    
    if group_by == 'both':
        return sorted_df.drop_duplicates(subset=['model', 'dataset'])
    else:
        return sorted_df.drop_duplicates(subset=[group_by])


def generate_summary_table(combined: pd.DataFrame) -> str:
    """
    Generate a markdown summary table of best results.
    
    Args:
        combined: Combined results DataFrame
    
    Returns:
        Markdown formatted string
    """
    if combined.empty:
        return "No results to summarize."
    
    # Get best result per model-dataset combination
    best_df = get_best_results(combined, 'both')
    
    # Select key columns that exist
    display_cols = ['model', 'dataset']
    for col in KEY_METRICS:
        if col in best_df.columns:
            display_cols.append(col)
    
    summary_df = best_df[display_cols].copy()
    
    # Format percentages
    pct_cols = [c for c in summary_df.columns if c in HIGHER_IS_BETTER]
    for col in pct_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    
    return summary_df.to_markdown(index=False)


def print_comparison_report(combined: pd.DataFrame):
    """
    Print a formatted comparison report.
    
    Args:
        combined: Combined results DataFrame
    """
    if combined.empty:
        return
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Best overall by dataset
    print("\n### Best Model per Dataset (by F1 Macro or Accuracy):")
    print("-" * 50)
    
    for dataset in combined['dataset'].unique():
        dataset_df = combined[combined['dataset'] == dataset]
        
        # Find best metric to use
        metric = 'f1_macro' if 'f1_macro' in dataset_df.columns else 'accuracy'
        if metric not in dataset_df.columns:
            continue
        
        best_idx = dataset_df[metric].idxmax()
        best_row = dataset_df.loc[best_idx]
        
        print(f"\n  {dataset.upper()}:")
        print(f"    Best Model: {best_row['model']}")
        
        for m in KEY_METRICS[:6]:  # Top 6 metrics
            if m in best_row.index and pd.notna(best_row[m]):
                if m in HIGHER_IS_BETTER:
                    print(f"    {m}: {best_row[m]:.4f}")
                else:
                    print(f"    {m}: {best_row[m]:.2f}")
    
    # Stage 2 models comparison (excluding isolation_forest)
    stage2_df = combined[combined['model'] != 'isolation_forest']
    if not stage2_df.empty:
        print("\n### Stage 2 Models Comparison:")
        print("-" * 50)
        
        # Pivot table for comparison
        metric = 'f1_macro' if 'f1_macro' in stage2_df.columns else 'accuracy'
        if metric in stage2_df.columns:
            try:
                # Get best result per model-dataset
                best_s2 = get_best_results(stage2_df, 'both', metric)
                pivot = best_s2.pivot_table(
                    index='dataset', 
                    columns='model', 
                    values=metric,
                    aggfunc='max'
                )
                print(f"\n{metric} by Dataset and Model:")
                print(pivot.round(4).to_string())
            except Exception as e:
                print(f"  Could not generate pivot table: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate and compare evaluation results across datasets"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for summary CSV (default: evaluation/summary.csv)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['csv', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Aggregate results
    combined = aggregate_results()
    
    if combined.empty:
        print("\nNo results to compare. Run experiments first.")
        sys.exit(1)
    
    # Print report
    if not args.quiet:
        print_comparison_report(combined)
    
    # Determine output path
    output_path = Path(args.output) if args.output else EVALUATION_DIR / 'summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    if args.format in ['csv', 'both']:
        combined.to_csv(output_path, index=False)
        print(f"\n✓ Saved full summary to {output_path}")
        
        # Best results per model-dataset
        best_results = get_best_results(combined, 'both')
        best_path = output_path.parent / 'summary_best.csv'
        best_results.to_csv(best_path, index=False)
        print(f"✓ Saved best results to {best_path}")
    
    if args.format in ['markdown', 'both']:
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write("# Cross-Dataset Evaluation Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Best Results per Dataset and Model\n\n")
            f.write(generate_summary_table(combined))
            f.write("\n")
        print(f"✓ Saved markdown summary to {md_path}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
