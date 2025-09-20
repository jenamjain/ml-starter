from model import train_model
from data import load_data

def test_model_training():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    # Check model is not None
    assert model is not None

    # Check accuracy is reasonable (at least 90%)
    accuracy = model.score(X_test, y_test)
    assert accuracy >= 0.9, f"Accuracy too low: {accuracy}"
