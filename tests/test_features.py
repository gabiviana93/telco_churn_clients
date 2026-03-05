from src.features import build_preprocessor

def test_build_preprocessor():
    num_features = ["age", "income"]
    cat_features = ["city"]

    preprocessor = build_preprocessor(num_features, cat_features)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2
