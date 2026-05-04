"""
Convert dae_examples/BouncingBallsN.mo into a DAE YAML specification.

The output YAML mirrors the structure of dae_examples/dae_specification_bouncing_balls.json
and is consumable by src/discrete_adjoint/dae_solver.DAESolver (use yaml.safe_load
instead of json.load when reading it).

Initial conditions are produced by Python translations of the Modelica functions
initialXUpperHalf / initialYUpperHalf / initialVx / initialVy.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import operator
import os
import random
import re
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Modelica parameter parser (lightweight, tailored to BouncingBallsN.mo).
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(
    r"""parameter\s+
        (Integer|Real)\s+         # type
        ([A-Za-z_]\w*)            # name
        (?:\s*\([^)]*\))?         # optional modifiers, e.g. (min=1)
        \s*=\s*
        ([^;"\n]+?)               # default-value expression (no ;, ", or newline)
        \s*(?=["\;])              # stopped by description string or ; terminator
    """,
    re.VERBOSE,
)

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_arith(expr: str) -> float:
    """Safely evaluate a Modelica numeric expression like '0.5*9.81' or '-2 + 3'."""
    tree = ast.parse(expr, mode="eval")

    def _e(node):
        if isinstance(node, ast.Expression):
            return _e(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            return _SAFE_BINOPS[type(node.op)](_e(node.left), _e(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
            return _SAFE_UNARYOPS[type(node.op)](_e(node.operand))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    return float(_e(tree))


def parse_modelica_parameters(mo_path: str) -> dict[str, float | int]:
    """Extract scalar `parameter Real|Integer name = <expr>` declarations.

    Right-hand sides may be arithmetic expressions over numeric literals
    (e.g. `0.5*9.81`, `-2 + 3`); they are evaluated safely. Declarations whose
    RHS references symbols (e.g. function calls or array initializers like
    `initialXUpperHalf(N, ...)`) are skipped.
    """
    with open(mo_path, "r") as fh:
        text = fh.read()

    out: dict[str, float | int] = {}
    for typ, name, raw_expr in _PARAM_RE.findall(text):
        expr = raw_expr.strip()
        try:
            val = _eval_arith(expr)
        except Exception:
            # Non-numeric RHS (e.g. `initialXUpperHalf(N, x_min, x_max)`) -- skip.
            continue
        out[name] = int(val) if typ == "Integer" else float(val)
    return out


# ---------------------------------------------------------------------------
# Translations of the Modelica algorithm functions.
# ---------------------------------------------------------------------------

def number_of_columns(N: int) -> int:
    n_cols = 1
    while n_cols * n_cols < N:
        n_cols += 1
    return n_cols


def number_of_rows(N: int, n_cols: int) -> int:
    return (N + n_cols - 1) // n_cols


def initial_x_upper_half(N: int, x_min: float, x_max: float) -> list[float]:
    n_cols = number_of_columns(N)
    out = []
    for i in range(1, N + 1):
        col = (i - 1) % n_cols + 1
        out.append(x_min + (col - 0.5) * (x_max - x_min) / n_cols)
    return out


def initial_y_upper_half(N: int, y_min: float, y_max: float) -> list[float]:
    n_cols = number_of_columns(N)
    n_rows = number_of_rows(N, n_cols)
    y_mid = y_min + 0.5 * (y_max - y_min)
    out = []
    for i in range(1, N + 1):
        row = (i - 1) // n_cols + 1
        out.append(y_mid + (row - 0.5) * (y_max - y_mid) / n_rows)
    return out


def initial_vx(N: int, vx_max0: float) -> list[float]:
    n_cols = number_of_columns(N)
    out = []
    for i in range(1, N + 1):
        col = (i - 1) % n_cols + 1
        if n_cols > 1:
            out.append(-vx_max0 + 2.0 * vx_max0 * (col - 1) / (n_cols - 1))
        else:
            out.append(0.0)
    return out


def initial_vy(N: int, vy_max0: float) -> list[float]:
    n_cols = number_of_columns(N)
    n_rows = number_of_rows(N, n_cols)
    out = []
    for i in range(1, N + 1):
        row = (i - 1) // n_cols + 1
        if n_rows > 1:
            out.append(-vy_max0 + 2.0 * vy_max0 * (row - 1) / (n_rows - 1))
        else:
            out.append(-vy_max0)
    return out


# ---------------------------------------------------------------------------
# DAE-spec builder.
# ---------------------------------------------------------------------------

def build_dae_spec(params: dict[str, float | int],
                   epsilon: float = 1.0e-4,
                   seed: int | None = 0) -> dict[str, Any]:
    """Return a dict matching the JSON spec schema (states/parameters/f/when/...).

    `epsilon` is the half-width of a uniform random perturbation added to every
    initial position and velocity, breaking ties so that no two balls share an
    exact x or y coordinate (or velocity component) at t=0. Set epsilon=0 to
    disable jitter. `seed` makes the perturbation reproducible.
    """
    N = int(params["N"])
    g = float(params.get("g", 9.81))
    e_g = float(params.get("e_g", 0.8))
    e_b = float(params.get("e_b", 0.9))
    d_sq = float(params.get("d_sq", 0.25))
    x_min = float(params.get("x_min", 0.0))
    x_max = float(params.get("x_max", 10.0))
    y_min = float(params.get("y_min", 0.0))
    y_max = float(params.get("y_max", 10.0))
    vx_max0 = float(params.get("vx_max0", 0.6))
    vy_max0 = float(params.get("vy_max0", 0.55))

    x0 = initial_x_upper_half(N, x_min, x_max)
    y0 = initial_y_upper_half(N, y_min, y_max)
    vx0 = initial_vx(N, vx_max0)
    vy0 = initial_vy(N, vy_max0)

    # Symmetry-breaking jitter: small uniform noise so that balls in the same
    # row/column don't share exact coordinates or velocity components, which
    # would otherwise cause simultaneous events and degenerate event ordering.
    if epsilon > 0.0:
        rng = random.Random(seed)
        def _jitter(values: list[float]) -> list[float]:
            return [v + rng.uniform(-epsilon, epsilon) for v in values]
        x0 = _jitter(x0)
        y0 = _jitter(y0)
        vx0 = _jitter(vx0)
        vy0 = _jitter(vy0)

    states: list[dict[str, Any]] = []
    for i in range(1, N + 1):
        for name, val in [
            (f"x{i}", x0[i - 1]),
            (f"y{i}", y0[i - 1]),
            (f"vx{i}", vx0[i - 1]),
            (f"vy{i}", vy0[i - 1]),
        ]:
            states.append({"name": name, "type": "float", "start": float(val), "orig_name": name})

    parameters = [
        {"name": "g",     "type": "float", "value": g,     "orig_name": "g"},
        {"name": "e_g",   "type": "float", "value": e_g,   "orig_name": "e_g"},
        {"name": "e_b",   "type": "float", "value": e_b,   "orig_name": "e_b"},
        {"name": "d_sq",  "type": "float", "value": d_sq,  "orig_name": "d_sq"},
        {"name": "y_max", "type": "float", "value": y_max, "orig_name": "y_max"},
        {"name": "y_min", "type": "float", "value": y_min, "orig_name": "y_min"},
        {"name": "x_max", "type": "float", "value": x_max, "orig_name": "x_max"},
        {"name": "x_min", "type": "float", "value": x_min, "orig_name": "x_min"},
    ]

    f: list[str] = []
    for i in range(1, N + 1):
        f += [
            f"der(x{i}) = vx{i}",
            f"der(y{i}) = vy{i}",
            f"der(vx{i}) = 0",
            f"der(vy{i}) = -g",
        ]

    when: list[dict[str, Any]] = []
    for i in range(1, N + 1):
        # Wall events: list-form reinit clamps the position to the wall in
        # addition to the velocity flip. Without the clamp, the state stays
        # slightly inside the trigger band after the event, which is a common
        # source of consecutive-event (Zeno) chattering.
        when.append({
            "comment": f"Floor B{i}",
            "condition": f"y{i} - y_min < 0",
            "reinit": [
                f"vy{i} + e_g * prev(vy{i}) = 0",
                f"y{i} - y_min = 0",
            ],
        })
        when.append({
            "comment": f"Ceiling B{i}",
            "condition": f"-y{i} + y_max < 0",
            "reinit": [
                f"vy{i} + e_g * prev(vy{i}) = 0",
                f"y{i} - y_max = 0",
            ],
        })
        when.append({
            "comment": f"Left Wall B{i}",
            "condition": f"x{i} - x_min < 0",
            "reinit": [
                f"vx{i} + e_g * prev(vx{i}) = 0",
                f"x{i} - x_min = 0",
            ],
        })
        when.append({
            "comment": f"Right Wall B{i}",
            "condition": f"-x{i} + x_max < 0",
            "reinit": [
                f"vx{i} + e_g * prev(vx{i}) = 0",
                f"x{i} - x_max = 0",
            ],
        })

    for i, j in itertools.combinations(range(1, N + 1), 2):
        when.append({
            "comment": f"Collision B{i}-B{j}",
            "condition": f"(x{i}-x{j})**2 + (y{i}-y{j})**2 - d_sq < 0",
            "reinit": [
                f"vx{i} + e_b * prev(vx{i}) = 0",
                f"vy{i} + e_b * prev(vy{i}) = 0",
                f"vx{j} + e_b * prev(vx{j}) = 0",
                f"vy{j} + e_b * prev(vy{j}) = 0",
            ],
        })

    return {
        "states": states,
        "alg_vars": [],
        "inputs": None,
        "outputs": None,
        "parameters": parameters,
        "f": f,
        "g": [],
        "h": [],
        "when": when,
    }


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------

def modelica_to_yaml(
    mo_path: str,
    output_yaml_path: str,
    N: int | None = None,
    overrides: dict[str, float | int] | None = None,
    epsilon: float = 1.0e-1,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Read a Modelica BouncingBallsN file and write a YAML DAE spec.

    `N` (if given) overrides the value parsed from the .mo file.
    `overrides` (optional) overrides any parameter (g, e_g, e_b, d_sq, x_min, ...).
    `epsilon` is a small symmetry-breaking jitter applied to every initial
    position and velocity (set to 0 to disable). `seed` controls reproducibility.
    Returns the DAE spec dict.
    """
    params = parse_modelica_parameters(mo_path)
    if N is not None:
        params["N"] = int(N)
    if overrides:
        params.update(overrides)
    if "N" not in params:
        raise ValueError(f"Could not find parameter N in {mo_path}; pass N= explicitly.")

    spec = build_dae_spec(params, epsilon=epsilon, seed=seed)

    with open(output_yaml_path, "w") as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return spec


def _main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    default_mo = os.path.join(root, "dae_examples", "BouncingBallsN.mo")

    ap = argparse.ArgumentParser(description="Modelica BouncingBallsN.mo -> DAE YAML spec.")
    ap.add_argument("--mo", default=default_mo, help="Path to BouncingBallsN.mo")
    ap.add_argument("--out", default=None,
                    help="Output YAML path (default: dae_examples/dae_specification_bouncing_balls_N<N>.yaml)")
    ap.add_argument("--N", type=int, default=None, help="Override N from the Modelica file")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="Symmetry-breaking jitter half-width on initial positions/velocities (default 1e-4; 0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the jitter (default 0)")
    args = ap.parse_args()

    params = parse_modelica_parameters(args.mo)
    N = args.N if args.N is not None else int(params.get("N", 25))
    out_path = args.out or os.path.join(
        root, "dae_examples", f"dae_specification_bouncing_balls_N{N}.yaml"
    )

    spec = modelica_to_yaml(args.mo, out_path, N=N,
                            epsilon=args.epsilon, seed=args.seed)
    n_states = len(spec["states"])
    n_eqs = len(spec["f"])
    n_when = len(spec["when"])
    print(f"Wrote {out_path}")
    print(f"  N={N}  states={n_states}  ODEs={n_eqs}  events={n_when} "
          f"(walls={4*N}, ball-pairs={n_when - 4*N})")


if __name__ == "__main__":
    _main()
