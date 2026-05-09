import warnings

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from paths import DATA_FILE, FEATURES, FIGURES_DIR, MODEL_FILE, RESULTS_DIR, ensure_output_dirs


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


warnings.filterwarnings("ignore")


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
    }


def save_roc_curve(models: dict[str, object], x_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, probabilities, name=model_name)

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title("ROC-AUC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_auc_curve.png", dpi=300)
    plt.close()


def save_feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    feature_importance = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance (%)": model.feature_importances_ * 100,
            }
        )
        .sort_values("Importance (%)", ascending=False)
        .reset_index(drop=True)
    )

    feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance (%)", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
    plt.close()

    return feature_importance


def save_confusion_matrix(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_random_forest.png", dpi=300)
    plt.close()


def save_model_metric_chart(metrics: pd.DataFrame) -> None:
    chart_df = metrics.melt(
        id_vars="Model",
        value_vars=["ROC-AUC", "Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=chart_df, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_performance_metrics.png", dpi=300)
    plt.close()


def save_dataset_summary(
    df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_resampled: pd.Series,
    feature_importance: pd.DataFrame,
    best_model_name: str,
) -> None:
    success_count = int(df["Success"].sum())
    total_count = len(df)
    success_rate = df["Success"].mean() * 100
    avg_gap_success = df[df["Success"] == 1]["Gap"].mean()
    avg_gap_fail = df[df["Success"] == 0]["Gap"].mean()
    top_feature = feature_importance.iloc[0]["Feature"]

    summary = "\n".join(
        [
            "# Dataset and Training Summary",
            "",
            f"Rows after dropna: {total_count}",
            f"Successful undercuts: {success_count} ({success_rate:.2f}%)",
            f"Unique races: {df['Race'].nunique()}",
            f"Unique drivers: {df['Driver'].nunique()}",
            f"Train class counts: {y_train.value_counts().to_dict()}",
            f"Validation class counts: {y_test.value_counts().to_dict()}",
            f"ADASYN class counts: {y_train_resampled.value_counts().to_dict()}",
            f"Average successful gap: {avg_gap_success:.3f} sec",
            f"Average failed gap: {avg_gap_fail:.3f} sec",
            f"Best validation model: {best_model_name}",
            f"Most important feature: {top_feature}",
            "",
        ]
    )
    (RESULTS_DIR / "dataset_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    print("=" * 64)
    print("F1 Undercut Prediction - Training Pipeline")
    print("=" * 64)

    df = pd.read_csv(DATA_FILE).dropna()
    x = df[FEATURES]
    y = df["Success"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    sampler = ADASYN(random_state=42)
    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)

    print(f"Rows after dropna: {len(df)}")
    print(f"Train class counts: {y_train.value_counts().to_dict()}")
    print(f"Validation class counts: {y_test.value_counts().to_dict()}")
    print(f"ADASYN class counts: {y_train_resampled.value_counts().to_dict()}")

    fitted_models = {}
    metric_rows = []

    for model_name, model in build_models().items():
        print(f"Training: {model_name}")
        model.fit(x_train_resampled, y_train_resampled)
        fitted_models[model_name] = model
        metric_rows.append({"Model": model_name, **evaluate_model(model, x_test, y_test)})

    metrics = pd.DataFrame(metric_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    best_model_name = metrics.iloc[0]["Model"]
    with (RESULTS_DIR / "classification_report_random_forest.txt").open("w", encoding="utf-8") as report_file:
        report_file.write(
            classification_report(
                y_test,
                fitted_models["Random Forest"].predict(x_test),
                zero_division=0,
            )
        )

    feature_importance = save_feature_importance(fitted_models["Random Forest"])
    save_roc_curve(fitted_models, x_test, y_test)
    save_confusion_matrix(fitted_models["Random Forest"], x_test, y_test)
    save_model_metric_chart(metrics)

    joblib.dump(fitted_models["Random Forest"], MODEL_FILE)
    save_dataset_summary(df, y_train, y_test, y_train_resampled, feature_importance, best_model_name)

    print("\nModel metrics:")
    print(metrics.to_string(index=False))
    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Results saved: {RESULTS_DIR}")
    print(f"Figures saved: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
