"""Self-contained CLI task package (see main.py + __main__.py)."""
from .main import CATEGORIES, generate_all, main

__all__ = ["CATEGORIES", "generate_all", "main"]
