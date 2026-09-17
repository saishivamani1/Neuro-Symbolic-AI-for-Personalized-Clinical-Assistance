#!/usr/bin/env python
"""
evaluate_symbolic_engine.py (Root Launcher)

Executes the automated 500-case Neuro-Symbolic Clinical Reasoning Benchmark.

Usage:
    python evaluate_symbolic_engine.py
"""

import os
import sys
from pathlib import Path

# Ensure backend is on sys.path
root_dir = Path(__file__).resolve().parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.evaluation.evaluate_symbolic_engine import run_evaluation

if __name__ == "__main__":
    out_dir = root_dir / "evaluation_results"
    run_evaluation(output_dir=out_dir)
