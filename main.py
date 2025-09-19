from data import get_data
from model import train_model

def main():
    X, y = get_data()
    model = train_model(X, y)
    print("Accuracy:", model.score(X, y))

if __name__ == "__main__":
    main()
