from src.preprocessing import split_data

def test_split_data_shapes(simple_test_dataframe):
    X_train, X_test, y_train, y_test = split_data(
        simple_test_dataframe, "target", test_size=0.5, random_state=42
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "target" not in X_train.columns
