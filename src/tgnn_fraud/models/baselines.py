"""One transaction-feature XGBoost baseline."""

import numpy as np
from xgboost import XGBClassifier


def train_baseline(
    features: np.ndarray, labels: np.ndarray, config: dict, seed: int
) -> XGBClassifier:
    """Fit on train features [N,F] and labels [N]; use train class weights only."""
    positives = int(labels.sum())
    if not 0 < positives < len(labels):
        raise ValueError("Baseline training requires both classes")
    model = XGBClassifier(
        **config,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=(len(labels) - positives) / positives,
    )
    model.fit(features, labels)
    return model
