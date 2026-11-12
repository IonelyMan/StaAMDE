import json
import os
import tempfile
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


class LightGBMBoosterModel:
    """A validation checkpoint usable by the existing inference and SHAP paths."""

    def __init__(self, model_string: str):
        self.model_string = model_string
        self._booster = None

    @property
    def booster_(self):
        if self._booster is None:
            import lightgbm as lgb
            self._booster = lgb.Booster(model_str=self.model_string)
        return self._booster

    def predict_proba(self, X):
        positive = np.asarray(self.booster_.predict(X), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def save_model_checkpoint(model, path: Path) -> None:
    """Replace a checkpoint only after the new joblib file was written completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lightgbm-", suffix=".joblib",
                                         delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_validation_checkpoint(booster, iteration, score, vectorizer, selected_indices,
                               selected_names, model_dir: Path) -> Path:
    path = model_dir / "best_checkpoint.joblib"
    payload = {
        "model": LightGBMBoosterModel(booster.model_to_string(num_iteration=iteration)),
        "vectorizer": vectorizer,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "best_iteration": iteration,
        "val_pr_auc": score,
    }
    save_model_checkpoint(payload, path)
    return path


def save_feature_artifacts(vectorizer, selected_indices, selected_names, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "dict_vectorizer.joblib")
    np.save(model_dir / "selected_indices.npy", selected_indices)
    (model_dir / "selected_features.json").write_text(
        json.dumps(selected_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_artifacts(
    model,
    vectorizer: DictVectorizer,
    selected_indices: np.ndarray,
    selected_names: List[str],
    model_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    save_feature_artifacts(vectorizer, selected_indices, selected_names, model_dir)
    save_model_checkpoint(model, model_dir / "lightgbm_model.joblib")
    (model_dir / "best_checkpoint.joblib").unlink(missing_ok=True)


def load_artifacts(model_dir: Path):
    checkpoint = model_dir / "best_checkpoint.joblib"
    if checkpoint.is_file():
        payload = joblib.load(checkpoint)
        return (payload["model"], payload["vectorizer"], payload["selected_indices"],
                payload["selected_names"])
    model = joblib.load(model_dir / "lightgbm_model.joblib")
    vectorizer = joblib.load(model_dir / "dict_vectorizer.joblib")
    selected_indices = np.load(model_dir / "selected_indices.npy")
    selected_names = json.loads((model_dir / "selected_features.json").read_text(encoding="utf-8"))
    return model, vectorizer, selected_indices, selected_names
