import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif


def binarize_for_mutual_info(X):
    if hasattr(X, "copy") and hasattr(X, "data"):
        X_binary = X.copy()
        X_binary.data = np.ones_like(X_binary.data, dtype=np.int8)
        return X_binary
    return (np.asarray(X) != 0).astype(np.int8)


def select_by_mutual_info(X_train, y_train, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X_mi = binarize_for_mutual_info(X_train)
    kwargs = {"discrete_features": True, "random_state": seed}
    try:
        scores = mutual_info_classif(X_mi, y_train, n_jobs=-1, **kwargs)
    except TypeError:
        scores = mutual_info_classif(X_mi, y_train, **kwargs)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if k <= 0 or k >= X_train.shape[1]:
        indices = np.arange(X_train.shape[1])
    else:
        indices = np.argsort(scores)[::-1][:k]
    return indices, scores


def predict_probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def save_artifacts(
    model,
    vectorizer: DictVectorizer,
    selected_indices: np.ndarray,
    selected_names: List[str],
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "lightgbm_model.joblib")
    joblib.dump(vectorizer, model_dir / "dict_vectorizer.joblib")
    np.save(model_dir / "selected_indices.npy", selected_indices)
    (model_dir / "selected_features.json").write_text(
        json.dumps(selected_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "lightgbm_model.joblib")
    vectorizer = joblib.load(model_dir / "dict_vectorizer.joblib")
    selected_indices = np.load(model_dir / "selected_indices.npy")
    selected_names = json.loads((model_dir / "selected_features.json").read_text(encoding="utf-8"))
    return model, vectorizer, selected_indices, selected_names
