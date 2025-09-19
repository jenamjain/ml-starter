from sklearn.datasets import load_iris

def get_data():
    """Loads the iris dataset and returns features and labels."""
    X, y = load_iris(return_X_y=True)
    return X, y
