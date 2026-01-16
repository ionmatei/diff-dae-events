"""
DEER-based DAE Optimization Module

This module provides optimization tools for DAE systems using the DEER
(Differentiate-Evaluate-Eliminate-Reuse) framework with BwdEulerDEER solver.
"""

from .dae_optimizer_deer import DAEOptimizerDEER

__all__ = ['DAEOptimizerDEER']
