from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline
    feature_cols: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]


def build_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    seed: int = 42,
) -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
    )

    # Strong, report-friendly baseline. Not overfitted; stable.
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",  # helps base rate imbalance + report discussion
    )

    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> TrainedModel:
    feature_cols = list(X_train.columns)

    categorical_cols = [c for c in feature_cols if X_train[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    pipe = build_pipeline(categorical_cols=categorical_cols, numeric_cols=numeric_cols, seed=seed)
    pipe.fit(X_train, y_train)

    return TrainedModel(
        pipeline=pipe,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


def predict_scores(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    # Positive class probability for >50K
    return model.pipeline.predict_proba(X)[:, 1]


def predict_labels(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_score))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
        "confusion_matrix": cm,  # [[TN, FP],[FN, TP]]
    }


def save_model(model: TrainedModel, path: str) -> None:
    dump(model.pipeline, path)
