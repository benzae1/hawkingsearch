"""Hawking search: a black-hole-evaporation metaheuristic."""
from . import baselines, problems
from .core import Budget, Result
from .full import FullConfig, MODULES
from .full import minimize as minimize_full
from .minimal import MinimalConfig
from .minimal import minimize as minimize_minimal

ALGORITHMS = {
    "hawking-minimal": minimize_minimal,
    "hawking-full": minimize_full,
    **baselines.ALL,
}

__all__ = ["ALGORITHMS", "Budget", "FullConfig", "MODULES", "MinimalConfig", "Result",
           "baselines", "minimize_full", "minimize_minimal", "problems"]
