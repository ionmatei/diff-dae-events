# Optimizer Benchmarking Suite

This directory contains scripts for benchmarking and comparing different DAE optimization approaches.

## Files

- **`benchmark_optimizers.py`** - Main benchmark script
- **`analyze_benchmarks.py`** - Analysis and visualization script
- **`results/`** - Directory where benchmark results are saved

## Quick Start

### 1. Run Benchmarks

Run the default benchmark suite (CPU only, ncp values: 500, 1000, 2000, 3000, 4000, 5000):

```bash
.venv/bin/python benchmark_optimizers.py
```

This will:
- Load settings from the config file (default: `config/config_cauer.yaml`)
- Use all parameters from config (algorithm, discretization_method, max_iterations, tol, etc.)
- Override only device and ncp for each benchmark run
- Test both DEER methods and Parallel Optimized optimizers
- Run 3 times per configuration for statistical significance
- Save results to `results/benchmark_YYYYMMDD_HHMMSS.json`

### 2. Analyze Results

After running benchmarks, analyze the latest results:

```bash
.venv/bin/python analyze_benchmarks.py
```

This will:
- Load the latest benchmark JSON file
- Print detailed statistics
- Generate performance comparison plots
- Save summary CSV and visualizations to `results/`

## Advanced Usage

### Custom Benchmark Configuration

**Use a different config file:**
```bash
.venv/bin/python benchmark_optimizers.py --config config/my_config.yaml
```

**Test specific ncp values:**
```bash
.venv/bin/python benchmark_optimizers.py --ncp 1000 2000 5000
```

**Test different discretization methods:**
```bash
.venv/bin/python benchmark_optimizers.py --methods trapezoidal bdf2 bdf6
```

**Test on GPU and CPU:**
```bash
.venv/bin/python benchmark_optimizers.py --devices cpu gpu
```

**Increase runs for better statistics:**
```bash
.venv/bin/python benchmark_optimizers.py --runs 5
```

**Full custom run:**
```bash
.venv/bin/python benchmark_optimizers.py \
    --config config/config_cauer.yaml \
    --ncp 500 1000 2000 \
    --methods bdf6 \
    --devices cpu \
    --runs 3 \
    --output results
```

### Analyze Specific Results

**Analyze a specific benchmark file:**
```bash
.venv/bin/python analyze_benchmarks.py --file results/benchmark_20260120_110000.json
```

**Show plots interactively:**
```bash
.venv/bin/python analyze_benchmarks.py --show
```

## Output Files

### Benchmark Results JSON

Results are saved as JSON files with the following structure:

```json
{
  "timestamp": "2026-01-20T11:00:00",
  "device": "cpu",
  "ncp_values": [500, 1000, 2000, 3000, 4000, 5000],
  "methods": ["bdf6"],
  "n_runs": 3,
  "benchmarks": [
    {
      "method": "DEER_methods",
      "discretization": "bdf6",
      "ncp": 500,
      "run": 0,
      "total_time": 5.234,
      "converged": true,
      "n_iterations": 20,
      "final_loss": 0.023,
      "avg_time_per_iter": 0.187,
      "success": true
    },
    ...
  ]
}
```

### Analysis Outputs

The analysis script generates:

1. **`benchmark_analysis.png`** - Comprehensive comparison plots:
   - Total time vs problem size
   - Iteration time vs problem size
   - Convergence quality comparison
   - Relative speedup analysis

2. **`benchmark_summary.csv`** - Summary statistics in CSV format for further analysis

3. **Console output** - Detailed statistics including:
   - Performance by method
   - Scaling analysis (time complexity)
   - Convergence statistics

## Benchmark Metrics

Each benchmark run captures:

- **`total_time`** - Total optimization time (seconds)
- **`n_iterations`** - Number of optimization iterations
- **`final_loss`** - Final loss value
- **`initial_loss`** - Initial loss value
- **`final_grad_norm`** - Final gradient norm
- **`avg_time_per_iter`** - Average time per iteration (seconds)
- **`min_time_per_iter`** - Minimum iteration time
- **`max_time_per_iter`** - Maximum iteration time
- **`converged`** - Whether optimization converged

## Methods Compared

1. **DEER Methods** (`DAEOptimizerDEERMethods`)
   - DEER fixed-point iteration
   - Companion matrix for BDF methods
   - O(log N) parallel scan depth

2. **Parallel Optimized** (`DAEOptimizerParallelOptimized`)
   - True BDF adjoint
   - Matrix-free VJP infrastructure
   - Sequential or parallel scan options

## Tips

- Start with smaller ncp values to verify setup
- Use `--runs 1` for quick testing
- CPU benchmarks are more consistent than GPU for timing
- Results include error bars showing variability across runs
- Check convergence rates to ensure fair comparison

## Example Workflow

```bash
# 1. Run quick test benchmark
.venv/bin/python benchmark_optimizers.py --ncp 500 1000 --runs 1

# 2. Analyze results
.venv/bin/python analyze_benchmarks.py --show

# 3. Run full benchmark suite
.venv/bin/python benchmark_optimizers.py

# 4. Generate final analysis
.venv/bin/python analyze_benchmarks.py
```
