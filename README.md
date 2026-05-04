# DAE sensitivity-free optimization

Parameter identification on hybrid (event-driven) differential-algebraic
systems via three gradient pipelines:

- **JAX discrete adjoint (`jax_da`)** — Sundials/IDA forward solve, then a
  segmented backward sweep with a sigmoid-blended loss
  ([src/discrete_adjoint/dae_padded_gradient.py](src/discrete_adjoint/dae_padded_gradient.py)).
- **JAX AD (`jax_ad`)** — diffrax forward integration with composite event
  detection and per-segment reinit, fully JIT-compiled
  ([src/dae/dae_optimizer_events.py](src/dae/dae_optimizer_events.py),
  [src/jax_baseline/bouncing_balls_n_jax.py](src/jax_baseline/bouncing_balls_n_jax.py)).
- **PyTorch AD (`pytorch_ad`)** — torchdiffeq forward integration with
  contact event handling
  ([src/pytorch/bouncing_balls_n.py](src/pytorch/bouncing_balls_n.py)).

This branch covers two benchmarks:

1. **Cauer ladder with a state-dependent reinit event** — stiff
   electrical circuit; compares `jax_ad` vs. `jax_da` on a single fixed
   model.
2. **Planar bouncing balls (N=3, 7, 15)** — contact-driven reinit
   events; compares all three methods across model sizes.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) and is pinned in
`uv.lock`.

```bash
uv sync
```

This creates `.venv/`. All commands below assume the project root as
working directory and use `.venv/bin/python` explicitly. The
discrete-adjoint path requires `scikits.odes` (Sundials/IDA), and the
JAX AD path can run on CPU or CUDA (`jax[cuda12]`).

## Repository layout

```
config/                      # YAML run configs (one per benchmark scenario)
dae_examples/                # DAE specs (.json/.yaml) and the source .mo file
src/
  dae/                       # event-aware diffrax DAE optimizer (jax_ad)
  discrete_adjoint/          # IDA forward + padded backward sweep (jax_da)
  jax_baseline/              # diffrax-based bouncing-balls baseline (jax_ad)
  pytorch/                   # torchdiffeq bouncing-balls model (pytorch_ad)
  run/                       # CLI entry points for both benchmarks
  benchmark/                 # multi-method/multi-N orchestration + plotting
results/                     # JSON records and figures
```

## Benchmark 1 — Cauer ladder with events

Config: [config/config_cauer_events.yaml](config/config_cauer_events.yaml)
Spec:   [dae_examples/dae_specification_cauer_events.json](dae_examples/dae_specification_cauer_events.json)

Run the JAX AD (diffrax) optimizer:

```bash
.venv/bin/python -m src.run.optimization_cauer_events
```

Run the JAX discrete-adjoint (IDA + padded sweep) optimizer:

```bash
.venv/bin/python -m src.run.optimization_cauer_events_da
```

Both runners read the same YAML and write to `results/`:

- AD:  `optimization_cauer_events_adam{,_loss,_outputs,_params}.{json,png}`
- DA:  `optimization_cauer_events_da{,_loss,_outputs,_params}.{json,png}`

### Flags — `optimization_cauer_events` (jax_ad)

| Flag | Description |
|---|---|
| `--config PATH` | Path to the events YAML config. Default: `config/config_cauer_events.yaml`. |
| `--spec PATH` | Override the DAE spec from `--config`. |
| `--diffrax-solver {Tsit5,Dopri5,Dopri8,Heun}` | Override `optimizer.diffrax_solver`. |
| `--rtol FLOAT` | Override `dae_solver.rtol`. |
| `--atol FLOAT` | Override `dae_solver.atol`. |
| `--max-segments INT` | Override `optimizer.max_segments`. |
| `--blend-sharpness FLOAT` | If > 0, sigmoid-blend the stage-3 segment selection (β; larger = sharper). Off by default for the AD runner — `optimizer.blend_sharpness` in the YAML is DA-only; AD reads `optimizer.ad_blend_sharpness` instead. Typical: 100–300. |
| `--bias-factor FLOAT` | Half-width of the log-uniform multiplicative bias on the initial parameters: `p_init = p_true · exp(factor · U(-1, 1))`. Default 0.1. |
| `--bias-seed INT` | RNG seed for the bias draw. Default 0. |
| `--max-iter-override INT` | Override `optimizer.max_iterations`. |
| `--results-dir PATH` | Output directory. Default `results/`. |
| `--no-plot` | Skip plot generation. |
| `--best-every INT` | If > 0, snapshot `(params, loss)` every N iters and return the best-loss snapshot at the end (instead of the last iterate). Default 5. |
| `--optimizer {adam,lbfgs}` | Outer optimizer. `lbfgs` ignores the YAML's `step_size`/`beta*`/`epsilon` and uses scipy L-BFGS-B. Default `adam`. |

### Flags — `optimization_cauer_events_da` (jax_da)

| Flag | Description |
|---|---|
| `--config PATH` | Path to the events YAML config (shared with the diffrax runner). |
| `--spec PATH` | Override the DAE spec from `--config`. |
| `--max-blocks INT` | Override `optimizer.max_blocks`. |
| `--max-points-per-segment INT` | Override `optimizer.max_points_per_segment`. |
| `--max-targets INT` | Override `optimizer.max_targets`. |
| `--blend-sharpness FLOAT` | Override `optimizer.blend_sharpness`. |
| `--bias-factor FLOAT` | See above. Default 0.1. |
| `--bias-seed INT` | See above. Default 0. |
| `--max-iter-override INT` | Override `optimizer.max_iterations`. |
| `--results-dir PATH` | Default `results/`. |
| `--no-plot` | Skip plot generation. |
| `--best-every INT` | Same semantics as the AD runner. Default 5. |

### AD vs. DA forward overlay

To overlay the AD and DA forward trajectories at the nominal parameters
(sanity check that both solvers agree before optimizing):

```bash
.venv/bin/python -m src.run.simulate_cauer_events_compare
# -> results/cauer_events_compare.png
```

Flags:

| Flag | Description |
|---|---|
| `--config PATH` | YAML config. Default `config/config_cauer_events.yaml`. |
| `--spec PATH` | Override the DAE spec from `--config`. |
| `--diffrax-solver {Tsit5,Dopri5,Dopri8,Heun}` | diffrax solver. |
| `--rtol FLOAT` / `--atol FLOAT` | Tolerances used by both integrators. |
| `--max-segments INT` | Static upper bound on diffrax event segments. |
| `--ncp-override INT` | Override the number of communication points. |
| `--stop-time-override FLOAT` | Override `stop_time`. |
| `--results-dir PATH` | Default `results/`. |
| `--prefix STR` | Output file prefix (under `--results-dir`). Default `cauer_events_compare`. |

## Benchmark 2 — Bouncing balls (N=3, 7, 15)

Configs:
- [config/config_bouncing_balls_N3.yaml](config/config_bouncing_balls_N3.yaml)
- [config/config_bouncing_balls_N7.yaml](config/config_bouncing_balls_N7.yaml)
- [config/config_bouncing_balls_N15.yaml](config/config_bouncing_balls_N15.yaml)

Specs (auto-generated from
[dae_examples/BouncingBallsN.mo](dae_examples/BouncingBallsN.mo)):
- `dae_examples/dae_specification_bouncing_balls_N{3,7,15}.yaml`

Run a single method on a single N (also produces a per-N optimization plot):

```bash
# JAX discrete adjoint
.venv/bin/python -m src.run.optimization_jax_bouncing_balls_N \
    --config config/config_bouncing_balls_N7.yaml

# PyTorch AD
.venv/bin/python -m src.run.optimization_pytorch_bouncing_balls_N \
    --config config/config_bouncing_balls_N7.yaml

# JAX AD (diffrax)
.venv/bin/python -m src.run.optimization_jax_baseline_bouncing_balls_N \
    --config config/config_bouncing_balls_N7.yaml
```

Each of the three single-method runners accepts only:

| Flag | Description |
|---|---|
| `--config PATH`, `-c PATH` | Path to the YAML config (relative to project root or absolute). Default: `config/config_bouncing_balls_N3.yaml`. |

All other knobs (number of iterations, tolerances, ADAM hyperparameters,
`max_blocks`, `max_segments`, etc.) come from the YAML — see the
`optimizer:` block of any of the three N configs.

### Full three-methods sweep across N=3, 7, 15

The benchmark driver runs all selected methods across all selected N
values and writes `results/benchmark_three_N{3,7,15}.json`:

```bash
.venv/bin/python -m src.benchmark.benchmark_three_methods_N
```

Flags:

| Flag | Description |
|---|---|
| `--results-dir PATH` | Output directory for the per-N JSONs. Default `results/`. |
| `--only LABELS`, `--N LABELS` | Comma-separated subset of `{N3, N7, N15}`. Default: all. |
| `--methods METHODS` | Comma-separated subset of `{jax_da, pytorch_ad, jax_ad}`. Default: all. |
| `--device {cpu,gpu,cuda}` | JAX device for `jax_da` and `jax_ad`. PyTorch is unaffected. Set **before** any JAX module is imported (the driver handles this). Default `cpu`. |

Examples:

```bash
# only N7, only jax_da and pytorch_ad
.venv/bin/python -m src.benchmark.benchmark_three_methods_N \
    --only N7 --methods jax_da,pytorch_ad

# run jax_da and jax_ad on GPU
.venv/bin/python -m src.benchmark.benchmark_three_methods_N --device gpu
```

After the JSONs exist, render the comparison figures (per-N: prediction
error vs. iteration, prediction error vs. wall-clock time, validation
loss):

```bash
.venv/bin/python -m src.benchmark.plot_three_methods_N
# -> results/benchmark_three_N{3,7,15}_prediction_error.png
#    results/benchmark_three_N{3,7,15}_prediction_error_vs_time.png
#    results/benchmark_three_N{3,7,15}_validation_loss.png
```

Flags:

| Flag | Description |
|---|---|
| `--results-dir PATH` | Directory containing the `benchmark_three_N{N}.json` files; PNGs are written to the same directory. Default `results/`. |
| `--only LABELS`, `--N LABELS` | Comma-separated subset of `{N3, N7, N15}`. Default: all that exist on disk. |
| `--methods METHODS` | Comma-separated subset of `{jax_da, pytorch_ad, jax_ad}`. Default: all available in each JSON. |
