"""NIDS tabular dataset loading and preprocessing.

Key properties:
- Explicit UNSW task definition: binary uses label; multiclass uses attack_cat.
- Leakage columns are always dropped before feature preprocessing.
- Preprocessors are fit on the training split only and then applied to test.
- Categorical columns are one-hot encoded instead of silently discarded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TabularDataset


UNSW_LEAK_COLUMNS = {"label", "attack_cat", "id"}
CICIDS_LEAK_COLUMNS = {
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Src IP",
    "Dst IP",
    "Timestamp",
}
COMMON_DROP_COLUMNS = {"Unnamed: 0", "index"}


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    df = _clean_column_names(df)
    return df


def infer_tabular_label_column(dataset_name: str, args) -> str:
    if getattr(args, "tabular_label_column", ""):
        return args.tabular_label_column.strip()
    if dataset_name == "cicids2017":
        return "Label"
    if dataset_name == "unsw_nb15":
        return "label" if args.tabular_task == "binary" else "attack_cat"
    return "Label"


def _default_train_test_paths(dataset_name: str, data_root: str | Path) -> Tuple[Path | None, Path | None, Path | None]:
    root = Path(data_root)
    if dataset_name == "unsw_nb15":
        candidates = [
            (
                root / "unsw" / "UNSW_NB15_training-set.csv",
                root / "unsw" / "UNSW_NB15_testing-set.csv",
            ),
            (
                root / "UNSW_NB15_training-set.csv",
                root / "UNSW_NB15_testing-set.csv",
            ),
        ]
        for train_path, test_path in candidates:
            if train_path.exists() and test_path.exists():
                return train_path, test_path, None
        all_path = root / "UNSW_NB15.csv"
        return None, None, all_path if all_path.exists() else None

    if dataset_name == "cicids2017":
        candidates = [
            (root / "cicids" / "cicids2017_train.csv", root / "cicids" / "cicids2017_test.csv"),
            (root / "cicids2017_train.csv", root / "cicids2017_test.csv"),
        ]
        for train_path, test_path in candidates:
            if train_path.exists() and test_path.exists():
                return train_path, test_path, None
        all_candidates = [
            root / "cicids" / "CICIDS2017.csv",
            root / "CICIDS2017.csv",
            root / "processed" / "cicids2017_binary.csv",
        ]
        for all_path in all_candidates:
            if all_path.exists():
                return None, None, all_path

    return None, None, None


def _resolve_frames(args, dataset_name: str, label_column: str, data_root: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if args.tabular_train_file and args.tabular_test_file:
        train_df = _read_dataframe(args.tabular_train_file)
        test_df = _read_dataframe(args.tabular_test_file)
        return train_df, test_df, "explicit_train_test"

    if args.tabular_all_file:
        all_df = _read_dataframe(args.tabular_all_file)
        if label_column not in all_df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {args.tabular_all_file}")
        train_df, test_df = train_test_split(
            all_df,
            test_size=args.tabular_test_split,
            stratify=all_df[label_column],
            random_state=args.seed,
        )
        return train_df, test_df, "single_file_random_split_debug_only"

    train_path, test_path, all_path = _default_train_test_paths(dataset_name, data_root)
    if train_path is not None and test_path is not None:
        return _read_dataframe(train_path), _read_dataframe(test_path), "default_train_test"

    if all_path is not None:
        all_df = _read_dataframe(all_path)
        if label_column not in all_df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {all_path}")
        train_df, test_df = train_test_split(
            all_df,
            test_size=args.tabular_test_split,
            stratify=all_df[label_column],
            random_state=args.seed,
        )
        return train_df, test_df, "default_single_file_random_split_debug_only"

    raise FileNotFoundError(
        "Could not find tabular dataset files. Provide --tabular-train-file and "
        "--tabular-test-file, or provide --tabular-all-file."
    )


def _drop_feature_columns(
    df: pd.DataFrame, dataset_name: str, label_column: str, extra_drop_columns: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available columns: {list(df.columns)}"
        )

    y = df[label_column].astype(str).fillna("UNKNOWN").to_numpy(dtype=object)
    drop_cols = set(COMMON_DROP_COLUMNS)
    drop_cols.add(label_column)
    drop_cols.update(str(c).strip() for c in extra_drop_columns)

    if dataset_name == "unsw_nb15":
        # Drop both target-related columns. For binary target=label, attack_cat leaks.
        # For multiclass target=attack_cat, label leaks normal-vs-attack information.
        drop_cols.update(UNSW_LEAK_COLUMNS)
    elif dataset_name == "cicids2017":
        # Label and identifier/time/IP columns should not be used as features.
        drop_cols.update(CICIDS_LEAK_COLUMNS)

    existing_drop = [c for c in drop_cols if c in df.columns]
    X_df = df.drop(columns=existing_drop, errors="ignore").copy()

    forbidden = set()
    if dataset_name == "unsw_nb15":
        forbidden.update(UNSW_LEAK_COLUMNS)
    elif dataset_name == "cicids2017":
        forbidden.add("Label")
    leaks = forbidden.intersection(set(X_df.columns))
    if leaks:
        raise AssertionError(f"Leakage columns still present in features: {sorted(leaks)}")

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    return X_df, y, sorted(existing_drop)


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _fit_transform_features(train_X_df: pd.DataFrame, test_X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    numeric_cols = train_X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_X_df.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns remain after leakage/identifier drops")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train = preprocessor.fit_transform(train_X_df)
    X_test = preprocessor.transform(test_X_df)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    feature_names: List[str] = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        try:
            cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
            feature_names.extend([str(x) for x in cat_names])
        except Exception:
            feature_names.extend(categorical_cols)

    return X_train.astype(np.float32), X_test.astype(np.float32), feature_names


def _encode_labels(y_train_raw: np.ndarray, y_test_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    label_values = sorted(set(y_train_raw.tolist()) | set(y_test_raw.tolist()))
    label_map = {str(v): i for i, v in enumerate(label_values)}
    y_train = np.array([label_map[str(v)] for v in y_train_raw], dtype=np.int64)
    y_test = np.array([label_map[str(v)] for v in y_test_raw], dtype=np.int64)
    return y_train, y_test, label_map


def load_nids_datasets(args, dataset_name: str, data_root: str | Path):
    """Load CICIDS2017 or UNSW_NB15 as CARES-Lite TabularDataset objects."""
    label_column = infer_tabular_label_column(dataset_name, args)
    train_df, test_df, split_source = _resolve_frames(args, dataset_name, label_column, data_root)

    train_X_df, y_train_raw, dropped_train = _drop_feature_columns(
        train_df,
        dataset_name=dataset_name,
        label_column=label_column,
        extra_drop_columns=args.tabular_drop_columns,
    )
    test_X_df, y_test_raw, dropped_test = _drop_feature_columns(
        test_df,
        dataset_name=dataset_name,
        label_column=label_column,
        extra_drop_columns=args.tabular_drop_columns,
    )

    # Align raw columns before fitting the preprocessor. Missing columns in test are filled with NaN.
    train_X_df, test_X_df = train_X_df.align(test_X_df, join="left", axis=1)

    X_train, X_test, feature_names = _fit_transform_features(train_X_df, test_X_df)
    y_train, y_test, label_map = _encode_labels(y_train_raw, y_test_raw)

    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)

    metadata = {
        "label_column": label_column,
        "tabular_task": getattr(args, "tabular_task", ""),
        "split_source": split_source,
        "num_features": int(X_train.shape[1]),
        "num_classes": int(len(label_map)),
        "label_map": label_map,
        "dropped_columns_train": dropped_train,
        "dropped_columns_test": dropped_test,
        "num_raw_feature_columns": int(train_X_df.shape[1]),
        "num_encoded_feature_columns": int(X_train.shape[1]),
        "feature_names_preview": feature_names[:20],
    }

    print("\n[NIDS] Dataset preprocessing summary")
    print(f"  dataset: {dataset_name}")
    print(f"  task: {metadata['tabular_task']}")
    print(f"  label_column: {label_column}")
    print(f"  split_source: {split_source}")
    print(f"  dropped_columns_train: {dropped_train}")
    print(f"  train shape: {X_train.shape}")
    print(f"  test shape: {X_test.shape}")
    print(f"  label_map: {label_map}")

    return train_dataset, test_dataset, int(X_train.shape[1]), int(len(label_map)), metadata
