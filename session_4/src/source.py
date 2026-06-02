import pandas as pd
from metadata import DATASETS_FOLDER


def load_data(file_name: str) -> pd.DataFrame:
    """Load the CSV dataset from the datasets folder."""
    return pd.read_csv(f"{DATASETS_FOLDER}/{file_name}")
