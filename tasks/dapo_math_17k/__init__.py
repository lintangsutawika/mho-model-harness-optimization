"""Self-contained CLI task package (see main.py + __main__.py)."""
from .main import (
    LANGUAGES,
    DEFAULT_DATASET,
    generate,
    generate_all,
    load_problem,
    load_problems,
    main,
)

__all__ = [
    "LANGUAGES",
    "DEFAULT_DATASET",
    "generate",
    "generate_all",
    "load_problem",
    "load_problems",
    "main",
]