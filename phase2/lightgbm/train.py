import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.common.metrics import binary_metrics
from phase2.common.splits import split_indices_auto
from phase2.lightgbm.dataset import load_static_train_val_dataset
from phase2.lightgbm.model import predict_probability, save_artifacts, select_by_mutual_info
from phase2.lightgbm.reports import write_feature_scores, write_json, write_lightgbm_importance


def lgb_pr_auc_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(set(y_true.tolist())) < 2:
        return "pr_auc", 0.0, True
    return "pr_auc", float(average_precision_score(y_true, y_pred)), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练静态特征 LightGBM 分类器")
    parser.add_argument("--input", required=True, help="phase1/static_feature_extract.py 输出目录")
    parser.add_argument("--output", required=True, help="训练运行目录，例如 runs/static_lgbm")
    parser.add_argument("--mi-k", type=int, default=5000,help="挑选的互信息特征数")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping", type=int, default=80)
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    parser.add_argument("--workers", type=int, default=4, help="静态 JSON 并行加载进程数；0 表示自动")
    parser.add_argument("--info", type=str, default="")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output)
    print("本次训练信息：",args.info)
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    for directory in [model_dir, metrics_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("缺少 lightgbm，请先安装 requirements.txt 中的依赖。") from exc

    print(f"加载数据集，workers={args.workers}")
    dataset = load_static_train_val_dataset(Path(args.input), workers=args.workers)
    print("数据集大小:", np.size(dataset.labels))
    train_idx, val_idx, test_idx, split_source = split_indices_auto(
        dataset.labels,
        dataset.splits,
        args.test_size,
        args.val_size,
        args.seed,
    )

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([dataset.features[int(index)] for index in train_idx])
    X = vectorizer.transform(dataset.features)
    feature_names = list(vectorizer.get_feature_names_out())
    selected_indices, mi_scores = select_by_mutual_info(X[train_idx], dataset.labels[train_idx], args.mi_k, args.seed)
    selected_names = [feature_names[int(index)] for index in selected_indices]
    X_selected = X[:, selected_indices]
    print("加载模型")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.9,
        colsample_bytree=0.85,
        random_state=args.seed,
        n_jobs=-1,
        class_weight=args.class_weight,
        metric="None",
    )
    print("开始训练模型")
    X_train = X_selected[train_idx]
    X_val = X_selected[val_idx]
    callbacks = []
    if args.early_stopping > 0 and len(set(dataset.labels[val_idx].tolist())) == 2:
        callbacks = [lgb.early_stopping(args.early_stopping), lgb.log_evaluation(50)]
        model.fit(
            X_train,
            dataset.labels[train_idx],
            eval_set=[(X_val, dataset.labels[val_idx])],
            eval_metric=lgb_pr_auc_metric,
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, dataset.labels[train_idx])
    print("模型训练完成")
    train_probs = predict_probability(model, X_selected[train_idx])
    val_probs = predict_probability(model, X_selected[val_idx])
    train_metrics = binary_metrics(dataset.labels[train_idx], train_probs, args.threshold)
    val_metrics = binary_metrics(dataset.labels[val_idx], val_probs, args.threshold)

    train_summary = {
        "train": train_metrics,
        "val": val_metrics,
        "n_samples": int(len(dataset.labels)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test_reserved": int(len(test_idx)),
        "n_features_raw": int(X.shape[1]),
        "n_features_selected": int(len(selected_indices)),
        "split_source": split_source,
        "threshold": float(args.threshold),
        "best_model_metric": "pr_auc",
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    print("保存相关文件")
    write_json(metrics_dir / "train_metrics.json", train_summary)
    write_json(reports_dir / "train_config.json", vars(args))
    write_feature_scores(reports_dir / "mutual_info_features.csv", feature_names, mi_scores, selected_indices)
    write_lightgbm_importance(reports_dir / "lightgbm_feature_importance.csv", model, selected_names)
    save_artifacts(model, vectorizer, selected_indices, selected_names, model_dir)
    print(json.dumps(train_summary["val"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
