"""Validated image or group folds for the microsuturing development set."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_folds(annotations, folds=5, seed=42, group_column=None):
    if folds < 2:
        raise ValueError("At least two folds are required.")
    required = {"Name", "safety_label"}
    if group_column:
        required.add(group_column)
    if not required.issubset(annotations.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(annotations.columns))}")
    if annotations[list(required)].isna().any().any():
        raise ValueError("Image names, score groups and split groups must not be missing.")
    names = annotations["Name"].astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("Image names must be nonempty and unique.")
    labels = annotations["safety_label"].to_numpy()
    if set(labels) != {0, 1, 2} or (pd.Series(labels).value_counts() < folds).any():
        raise ValueError("Each of the three score groups needs at least one sample per fold.")
    positions = np.arange(len(annotations))
    groups = None
    if group_column:
        groups = annotations[group_column].astype(str).str.strip().to_numpy()
        if (groups == "").any() or len(set(groups)) < folds:
            raise ValueError("Provide nonempty groups and at least as many groups as folds.")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        candidates = splitter.split(positions, labels, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        candidates = splitter.split(positions, labels)
    result = []
    for train_indices, validation_indices in candidates:
        if set(labels[train_indices]) != {0, 1, 2} or set(labels[validation_indices]) != {0, 1, 2}:
            raise ValueError("A fold lacks a score group; reduce folds or revise the group design.")
        if groups is not None and set(groups[train_indices]) & set(groups[validation_indices]):
            raise ValueError("A group crosses training and validation boundaries.")
        result.append((train_indices, validation_indices))
    return result
