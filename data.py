from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    """Loads the iris dataset and returns train-test split."""
    X, y = load_iris(return_X_y=True)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
