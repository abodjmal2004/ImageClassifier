"""Train and evaluate the cats-versus-dogs SVM classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from utils import (
    load_and_preprocess_data,
    save_confusion_matrix,
    save_metrics,
    save_roc_curve,
)

CATEGORIES = ["cat", "dog"]
IMAGE_SIZE = (40, 40)
RANDOM_STATE = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an SVM model for binary cat/dog image classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/catsAndDogs40"),
        help="Dataset root containing category folders or train/test splits.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/svm_model.joblib"),
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation figures and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--cv", type=int, default=3, help="Number of cross-validation folds.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small parameter grid for a faster demonstration run.",
    )
    return parser


def train(args: argparse.Namespace) -> dict:
    print("Loading and preprocessing images...")
    X, y = load_and_preprocess_data(args.data_dir, CATEGORIES, IMAGE_SIZE)
    if len(set(y.tolist())) < len(CATEGORIES):
        raise ValueError("The dataset must contain at least one image for each category.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    parameter_grid = (
        {"estimator__C": [1, 10], "estimator__gamma": ["scale"], "estimator__kernel": ["rbf"]}
        if args.quick
        else {
            "estimator__C": [0.1, 1, 10, 100],
            "estimator__gamma": ["scale", 0.001, 0.01],
            "estimator__kernel": ["rbf", "poly"],
        }
    )
    search = GridSearchCV(
        estimator=CalibratedClassifierCV(
            estimator=SVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=3,
        ),
        param_grid=parameter_grid,
        cv=args.cv,
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
    )

    print("Training SVM and selecting the best parameters...")
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    probabilities = search.predict_proba(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=range(len(CATEGORIES)),
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, target_names=CATEGORIES, zero_division=0))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(
        y_test,
        predictions,
        CATEGORIES,
        args.results_dir / "confusion_matrix.png",
    )
    if len(set(y_test.tolist())) == 2:
        save_roc_curve(y_test, probabilities, args.results_dir / "roc_curve.png")

    metrics = {
        "accuracy": accuracy,
        "best_params": search.best_params_,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "image_size": list(IMAGE_SIZE),
        "classification_report": report,
    }
    save_metrics(metrics, args.results_dir / "metrics.json")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": search.best_estimator_,
            "categories": CATEGORIES,
            "image_size": IMAGE_SIZE,
            "format_version": 1,
        },
        args.model_path,
    )
    print(f"Saved model bundle to {args.model_path}")
    print(f"Saved evaluation artifacts to {args.results_dir}")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
