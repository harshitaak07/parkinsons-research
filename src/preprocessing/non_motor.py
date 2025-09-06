def load_non_motor_json(path: str) -> pd.DataFrame:
    data = pd.read_json(path, lines=True)
    return data

def clean_non_motor(df: pd.DataFrame) -> pd.DataFrame:
    return df
