import joblib
import pandas as pd

from paths import DATA_FILE, FEATURES, MODEL_FILE, REPORTS_DIR, RESULTS_DIR, ensure_output_dirs


def strategy_comment(most_important_feature: str) -> str:
    comments = {
        "Tyre_Advantage": "Undercut basarisinda lastik avantaji belirleyici faktor olarak one cikiyor.",
        "Ahead_TyreLife": "Rakibin lastik asinmasi en kritik firsat sinyalini uretiyor.",
        "Gap": "Araclar arasindaki saniye farki undercut basarisini dogrudan etkiliyor.",
        "Driver_TyreLife": "Undercut yapan pilotun lastik durumu karar kalitesini belirgin etkiliyor.",
    }
    return comments.get(most_important_feature, "Model, birden fazla stratejik degiskeni birlikte degerlendiriyor.")


def main() -> None:
    ensure_output_dirs()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found. Run `python src/train_model.py` first: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE).dropna()

    df["AI_Probability_%"] = (model.predict_proba(df[FEATURES])[:, 1] * 100).round(2)

    output_columns = ["Race", "Driver", "Car_Ahead", "Gap", "Tyre_Advantage", "AI_Probability_%", "Success"]
    top_undercuts = df.sort_values("AI_Probability_%", ascending=False).head(5)
    real_successes = df[df["Success"] == 1].sort_values("AI_Probability_%", ascending=False).head(5)

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

    most_important = feature_importance.iloc[0]["Feature"]
    avg_gap_success = df[df["Success"] == 1]["Gap"].mean()
    avg_gap_fail = df[df["Success"] == 0]["Gap"].mean()
    high_probability_count = int((df["AI_Probability_%"] > 40).sum())

    top_undercuts[output_columns].to_csv(RESULTS_DIR / "top_undercut_scenarios.csv", index=False)
    real_successes[output_columns].to_csv(RESULTS_DIR / "real_success_examples.csv", index=False)

    report = "\n".join(
        [
            "# F1 Undercut Strategy Analysis Report",
            "",
            "## Highest Probability Scenarios",
            top_undercuts[output_columns].to_markdown(index=False),
            "",
            "## Real Successful Examples",
            real_successes[output_columns].to_markdown(index=False),
            "",
            "## Feature Importance",
            feature_importance.to_markdown(index=False),
            "",
            "## Strategic Findings",
            f"- Average successful undercut gap: {avg_gap_success:.3f} sec",
            f"- Average failed undercut gap: {avg_gap_fail:.3f} sec",
            f"- Most important feature: {most_important}",
            f"- Scenarios above 40% AI probability: {high_probability_count}",
            f"- Model comment: {strategy_comment(most_important)}",
            "",
        ]
    )

    report_path = REPORTS_DIR / "AI_Strategy_Report.md"
    report_path.write_text(report, encoding="utf-8")

    print("Strategy report generated.")
    print(f"Report: {report_path}")
    print(f"Scenario outputs: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
