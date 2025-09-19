from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    """Trains a Logistic Regression model on given data."""
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model
