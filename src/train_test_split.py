
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data_cfg: dict, split_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:

    data_dir = Path(data_cfg['data_path'])
    data = pd.read_csv(data_dir / data_cfg["raw_data_file"])
    print(f"Data imported: {data.shape}")

    train_df, test_df = train_test_split(
        data,
        test_size=split_cfg["test_size"],
        stratify=data[split_cfg["stratify_on"]],
        random_state=split_cfg["random_state"],
    )

    train_path = data_dir / data_cfg["raw_train_data"]
    test_path = data_dir / data_cfg["raw_test_data"]
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_df.shape} -> {train_path}")
    print(f"Test: {test_df.shape} -> {test_path}")
    return train_df, test_df