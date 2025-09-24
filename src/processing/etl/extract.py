import pandas as pd
from pathlib import Path


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(p)

