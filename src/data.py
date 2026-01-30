from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import pandas as pd


ADULT_COLUMNS: List[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@dataclass(frozen=True)
class AdultDataset:
    train: pd.DataFrame
    test: pd.DataFrame


def _read_adult_file(path: Path, is_test: bool) -> pd.DataFrame:
    """
    Reads adult.data or adult.test into a DataFrame with consistent columns.
    Handles:
    - comma-separated values
    - extra '.' at end of label in adult.test
    - leading/trailing spaces
    - missing values indicated by '?'
    """
    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values=["?"],
        skipinitialspace=True,
        comment="|",
    )

    # adult.test has a header line and labels like '>50K.' / '<=50K.'
    if is_test:
        # Some distributions include a first line like: "|1x3 Cross validator"
        # If it parsed into NaNs, drop fully empty rows
        df = df.dropna(how="all")
        df["income"] = df["income"].astype(str).str.replace(".", "", regex=False)

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


def load_adult(data_dir: str | Path) -> AdultDataset:
    data_dir = Path(data_dir)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    train = _read_adult_file(train_path, is_test=False)
    test = _read_adult_file(test_path, is_test=True)

    # Standardize income labels
    # Expected: <=50K and >50K
    train["income"] = train["income"].replace({"<=50K.": "<=50K", ">50K.": ">50K"})
    test["income"] = test["income"].replace({"<=50K.": "<=50K", ">50K.": ">50K"})

    return AdultDataset(train=train, test=test)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["income"]).copy()
    y = (df["income"] == ">50K").astype(int)
    return X, y
