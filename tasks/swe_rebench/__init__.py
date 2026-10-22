"""Self-contained CLI task package (see main.py + __main__.py)."""
from .main import (
    DATASET_NAME,
    DEFAULT_DATASET,
    build_dataset,
    fetch_parsers,
    generate,
    generate_all,
    load_problem,
    load_problems,
    main,
)

__all__ = [
    "DATASET_NAME",
    "DEFAULT_DATASET",
    "build_dataset",
    "fetch_parsers",
    "generate",
    "generate_all",
    "load_problem",
    "load_problems",
    "main",
]
