import pandas as pd

from src.evaluate import evaluate_bundle, save_evaluation_artifacts
from src.model_bundle import CreditScoringModel
from src.preprocessing import make_logistic_pipeline


def test_evaluation_saves_metrics_and_figures(tmp_path):
    modeling_data = pd.DataFrame(
        {
            "SK_ID_CURR": list(range(1, 13)),
            "TARGET": [0, 0, 0, 1, 1, 1] * 2,
            "amount": [
                10.0,
                20.0,
                30.0,
                70.0,
                80.0,
                90.0,
                15.0,
                25.0,
                35.0,
                75.0,
                85.0,
                95.0,
            ],
        }
    )
    split = pd.DataFrame(
        {
            "SK_ID_CURR": list(range(1, 13)),
            "split": ["train"] * 6 + ["holdout"] * 6,
        }
    )
    estimator = make_logistic_pipeline()
    estimator.fit(
        modeling_data.loc[:5, ["amount"]],
        modeling_data.loc[:5, "TARGET"],
    )
    bundle = CreditScoringModel(
        estimator=estimator,
        feature_names=("amount",),
        model_type="logistic",
    )

    metrics, evaluation = evaluate_bundle(bundle, modeling_data, split)
    tables = tmp_path / "reports" / "tables"
    figures = tmp_path / "reports" / "figures"
    save_evaluation_artifacts(
        bundle,
        metrics,
        evaluation,
        tables_dir=tables,
        figures_dir=figures,
    )

    assert (tables / "final_evaluation.csv").is_file()
    assert (tables / "confusion_matrix.csv").is_file()
    assert (tables / "risk_segments.csv").is_file()
    assert (figures / "confusion_matrix.png").is_file()
    assert (figures / "calibration_curve.png").is_file()
    assert (figures / "score_distribution.png").is_file()
