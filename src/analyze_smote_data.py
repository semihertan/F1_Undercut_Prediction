import warnings

import matplotlib
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from paths import DATA_FILE, FEATURES, FIGURES_DIR, RESULTS_DIR, ensure_output_dirs


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


warnings.filterwarnings("ignore")


def summarize(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        rows.append(
            {
                "Feature": feature,
                "Original Mean": original[feature].mean(),
                "Original Std": original[feature].std(),
                "Synthetic Mean": synthetic[feature].mean(),
                "Synthetic Std": synthetic[feature].std(),
                "Synthetic Min": synthetic[feature].min(),
                "Synthetic Max": synthetic[feature].max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_output_dirs()

    df = pd.read_csv(DATA_FILE).dropna()
    x = df[FEATURES]
    y = df["Success"]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_train, y_train)
    original_size = len(x_train)

    original_success = x_train[y_train == 1].copy()
    original_success["Type"] = "Original"

    synthetic = x_resampled.iloc[original_size:].copy()
    synthetic_y = y_resampled.iloc[original_size:]
    synthetic_success = synthetic[synthetic_y == 1].copy()
    synthetic_success["Type"] = "Synthetic (SMOTE)"

    combined = pd.concat([original_success, synthetic_success], ignore_index=True)
    summary = summarize(original_success, synthetic_success)
    summary.to_csv(RESULTS_DIR / "smote_synthetic_summary.csv", index=False)

    plt.figure(figsize=(14, 10))
    for index, feature in enumerate(FEATURES, 1):
        plt.subplot(2, 2, index)
        sns.kdeplot(
            data=combined,
            x=feature,
            hue="Type",
            fill=True,
            common_norm=False,
            alpha=0.5,
            palette=["#1f77b4", "#2ca02c"],
        )
        plt.title(f"{feature} - Original vs SMOTE")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    figure_path = FIGURES_DIR / "smote_istatistik_analizi.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("SMOTE analysis completed.")
    print(f"Summary: {RESULTS_DIR / 'smote_synthetic_summary.csv'}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
