"""
Analyze and visualize benchmark results from benchmark_optimizers.py

This script loads benchmark JSON files and creates visualizations where:
- Each metric is shown in a separate subplot
- Lines represent each combination of (device, method, ncp)
- Missing metrics are handled gracefully
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import pandas as pd


def load_benchmark_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_dataframe(results: Dict) -> pd.DataFrame:
    """Convert benchmark results to pandas DataFrame."""
    # Filter successful benchmarks only
    successful = [b for b in results['benchmarks'] if b.get('success', False)]
    
    if not successful:
        raise ValueError("No successful benchmarks found in results!")
    
    df = pd.DataFrame(successful)
    return df


def get_combination_label(row):
    """Create label for a specific combination of device, method, and other attributes."""
    method = row['method']
    device = row.get('device', 'unknown')
    
    # Get scan_mode if available
    scan_mode = row.get('scan_mode', '')
    if pd.isna(scan_mode):
        scan_mode = ''
    
    label_parts = [method, device.upper()]
    
    # Add discretization if available
    discretization = row.get('discretization', '')
    if not pd.isna(discretization) and discretization:
        label_parts.append(discretization)

    if scan_mode:
        label_parts.append(scan_mode)
    
    return ' / '.join(label_parts)


def create_metric_figure(df: pd.DataFrame, title_suffix: str = ""):
    """
    Create a figure with each metric as a separate subplot.
    Each line represents a (device, method) combination across different NCP values.
    Returns the figure object.
    """
    
    # Define metrics to plot (key: column name, value: (ylabel, title, log_scale))
    metrics = {
        'avg_time_per_iter': ('Time (ms)', 'Average Iteration Time', False, 1000),
        'rel_error': ('Relative Error', 'Parameter Accuracy', True, 1),
        'final_loss': ('Loss', 'Final Loss', True, 1),
        'total_time': ('Time (s)', 'Total Optimization Time', False, 1),
    }
    
    # Optional metrics (only plot if available)
    optional_metrics = {
        'avg_dae_solve_time': ('Time (ms)', 'DAE Solve Time (discrete_adjoint only)', False, 1000),
        'avg_adjoint_time': ('Time (ms)', 'Adjoint Time (discrete_adjoint only)', False, 1000),
    }
    
    # Count how many metrics we'll actually plot
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    available_optional = {k: v for k, v in optional_metrics.items() if k in df.columns}
    
    n_metrics = len(available_metrics) + len(available_optional)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Get unique combinations of (device, method)
    # Get unique combinations
    group_cols = ['device', 'method']
    if 'discretization' in df.columns:
        group_cols.append('discretization')
        
    if 'device' in df.columns:
        combinations = df.groupby(group_cols).size().index.tolist()
    else:
        # Handle case with no device column
        group_cols = ['method']
        if 'discretization' in df.columns:
            group_cols.append('discretization')
        combinations = df.groupby(group_cols).apply(lambda x: ('cpu', *x.name) if isinstance(x.name, tuple) else ('cpu', x.name)).tolist()
    
    plot_idx = 0
    
    # Plot primary metrics
    for metric_name, (ylabel, title, log_scale, multiplier) in available_metrics.items():
        ax = axes[plot_idx]
        
        for combo in combinations:
            # Handle variable length combinations
            if len(combo) == 3:
                device, method, discretization = combo
                combo_data = df[(df['device'] == device) & (df['method'] == method) & (df['discretization'] == discretization)]
            elif len(combo) == 2:
                # Could be (device, method) or (method, discretization) depending on cols
                if 'discretization' in group_cols and 'device' not in group_cols:
                     method, discretization = combo
                     device = 'cpu'
                     combo_data = df[(df['method'] == method) & (df['discretization'] == discretization)]
                else:
                    device, method = combo
                    combo_data = df[(df['device'] == device) & (df['method'] == method)]
            else:
                method = combo[0] if isinstance(combo, tuple) else combo
                combo_data = df[df['method'] == method]
                device = 'cpu'
            
            if len(combo_data) == 0:
                continue
            
            # Get representative row for labeling
            rep_row = combo_data.iloc[0]
            label = get_combination_label(rep_row)
            
            # Group by ncp and compute mean/std
            grouped = combo_data.groupby('ncp')[metric_name].agg(['mean', 'std', 'count'])
            
            ax.errorbar(grouped.index, grouped['mean'] * multiplier, 
                       yerr=grouped['std'] * multiplier,
                       marker='o', linewidth=2, capsize=5, label=label, markersize=8)
        
        ax.set_xlabel('NCP', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot optional metrics (only for methods that have them)
    for metric_name, (ylabel, title, log_scale, multiplier) in available_optional.items():
        ax = axes[plot_idx]
        
        # Only plot for combinations that have this metric
        # Only plot for combinations that have this metric
        for combo in combinations:
            # Handle variable length combinations
            if len(combo) == 3:
                device, method, discretization = combo
                combo_data = df[(df['device'] == device) & (df['method'] == method) & (df['discretization'] == discretization)]
            elif len(combo) == 2:
                if 'discretization' in group_cols and 'device' not in group_cols:
                     method, discretization = combo
                     device = 'cpu'
                     combo_data = df[(df['method'] == method) & (df['discretization'] == discretization)]
                else:
                    device, method = combo
                    combo_data = df[(df['device'] == device) & (df['method'] == method)]
            else:
                method = combo[0] if isinstance(combo, tuple) else combo
                combo_data = df[df['method'] == method]
                device = 'cpu'
            
            # Check if this combination has the metric
            if metric_name not in combo_data.columns or combo_data[metric_name].isna().all():
                continue
            
            rep_row = combo_data.iloc[0]
            label = get_combination_label(rep_row)
            
            grouped = combo_data.groupby('ncp')[metric_name].agg(['mean', 'std'])
            
            ax.errorbar(grouped.index, grouped['mean'] * multiplier,
                       yerr=grouped['std'] * multiplier,
                       marker='s', linewidth=2, capsize=5, label=label, markersize=8)
        
        ax.set_xlabel('NCP', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
        
    if title_suffix:
        fig.suptitle(f"Benchmark Results - {title_suffix}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if title_suffix:
        plt.subplots_adjust(top=0.92)
    
    return fig


def print_statistics(df: pd.DataFrame):
    """Print detailed statistics from benchmark results."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal successful runs: {len(df)}")
    
    # Show devices and NCPs tested
    if 'device' in df.columns:
        devices = sorted(df['device'].unique())
        print(f"Devices tested: {devices}")
    ncp_values = sorted(df['ncp'].unique())
    print(f"NCP values tested: {ncp_values}")
    print(f"Methods compared: {list(df['method'].unique())}")
    
    # Group results by device and method
    print("\n" + "-" * 80)
    print("RESULTS BY METHOD AND DEVICE")
    print("-" * 80)
    
    for method in sorted(df['method'].unique()):
        method_data = df[df['method'] == method]
        
        print(f"\n{method.upper()}:")
        
        if 'device' in df.columns:
            for device in sorted(method_data['device'].unique()):
                device_method_data = method_data[method_data['device'] == device]
                print(f"\n  Device: {device.upper()}")
                
                if 'discretization' in df.columns:
                    for disc in sorted(device_method_data['discretization'].unique()):
                        disc_data = device_method_data[device_method_data['discretization'] == disc]
                        print(f"    Discretization: {disc}")
                        print_method_stats(disc_data, indent="      ")
                else:
                    print_method_stats(device_method_data, indent="    ")
        else:
            if 'discretization' in df.columns:
                for disc in sorted(method_data['discretization'].unique()):
                    disc_data = method_data[method_data['discretization'] == disc]
                    print(f"  Discretization: {disc}")
                    print_method_stats(disc_data, indent="    ")
            else:
                print_method_stats(method_data, indent="  ")


def print_method_stats(data: pd.DataFrame, indent: str = ""):
    """Print statistics for a specific method/device combination."""
    print(f"{indent}Avg iteration time: {data['avg_time_per_iter'].mean()*1000:.2f}ms ± {data['avg_time_per_iter'].std()*1000:.2f}ms")
    print(f"{indent}Relative error:     {data['rel_error'].mean():.6e} ± {data['rel_error'].std():.6e}")
    print(f"{indent}Final loss:         {data['final_loss'].mean():.6e} ± {data['final_loss'].std():.6e}")
    
    # Show timing breakdown if available
    if 'avg_dae_solve_time' in data.columns and not data['avg_dae_solve_time'].isna().all():
        dae_time = data['avg_dae_solve_time'].mean() * 1000
        adj_time = data['avg_adjoint_time'].mean() * 1000
        dae_fraction = (data['avg_dae_solve_time'].mean() / data['avg_time_per_iter'].mean()) * 100
        print(f"{indent}DAE solve time:     {dae_time:.2f}ms ({dae_fraction:.1f}%)")
        print(f"{indent}Adjoint time:       {adj_time:.2f}ms")


def generate_report(filepath: str, output_dir: str = 'results'):
    """Generate complete analysis report from benchmark results."""
    
    print(f"\nLoading results from: {filepath}")
    results = load_benchmark_results(filepath)
    
    print(f"Benchmark timestamp: {results['timestamp']}")
    
    # Handle both old and new format (device vs devices)
    if 'devices' in results:
        print(f"Devices tested: {results['devices']}")
    elif 'device' in results:
        print(f"Device: {results['device']}")
    
    # Create DataFrame
    df = create_dataframe(results)
    
    # Print statistics
    print_statistics(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Check if we have discretization column to split by
    if 'discretization' in df.columns and not df['discretization'].isna().all():
        discretizations = sorted([d for d in df['discretization'].unique() if d])
        
        for disc in discretizations:
            print(f"  Generating plot for {disc}...")
            # Filter data for this discretization
            disc_df = df[df['discretization'] == disc]
            
            fig = create_metric_figure(disc_df, title_suffix=disc)
            
            # Save figure
            output_path = Path(output_dir) / f'benchmark_analysis_{disc}.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {output_path}")
            plt.close(fig)
            
        # Also create a combined plot just in case
        print("  Generating combined plot...")
        fig = create_metric_figure(df, title_suffix="All Methods")
        output_path = Path(output_dir) / 'benchmark_analysis_all.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close(fig)
        
    else:
        # Legacy behavior - single plot
        fig = create_metric_figure(df)
        output_path = Path(output_dir) / 'benchmark_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Save summary CSV
    summary_path = Path(output_dir) / 'benchmark_summary.csv'
    df.to_csv(summary_path, index=False)
    print(f"\nFull results CSV saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--file', '-f', type=str,
                       help='Benchmark JSON file to analyze (if not specified, uses latest)')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Find benchmark file
    if args.file:
        filepath = args.file
    else:
        # Find latest benchmark file
        results_dir = Path(args.output)
        benchmark_files = list(results_dir.glob('benchmark_*.json'))
        
        if not benchmark_files:
            print(f"Error: No benchmark files found in {results_dir}")
            print("Run benchmark_optimizers.py first to generate results.")
            exit(1)
        
        filepath = str(sorted(benchmark_files)[-1])
        print(f"Using latest benchmark file: {filepath}")
    
    # Generate report
    generate_report(filepath, output_dir=args.output)
    
    # Show plots if requested
    if args.show:
        plt.show()
