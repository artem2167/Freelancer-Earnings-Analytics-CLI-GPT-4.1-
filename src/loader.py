import pandas as pd
from config import DATA_PATH

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Приведём названия колонок к lower_case, уберём пробелы:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Приведём числовые поля к нужным типам:
    num_cols = ["earnings_usd", "job_duration_days", "project_type_fixed", "job_completed"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
