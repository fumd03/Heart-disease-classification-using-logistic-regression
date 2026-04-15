import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, test_size, random_state):
    path = "data/heart.csv"
    df = pd.read_csv(path)

    # Features (ALL columns except target)
    X = df.drop("target", axis=1)

    # Target
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
