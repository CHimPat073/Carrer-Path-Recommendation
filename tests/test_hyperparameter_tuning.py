from ml.training.hyperparameter_tuning import build_model_specs


def test_model_specs_cover_required_models():
    specs = build_model_specs()

    assert set(specs.keys()) == {"random_forest", "xgboost", "lightgbm"}
    assert all(spec["model_name"] for spec in specs.values())
    assert all(spec["search_space"] for spec in specs.values())
