from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Train logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Print accuracy
print("Accuracy:", model.score(X, y))
