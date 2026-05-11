# src/utils.py
# save the model to disk and load the model from disk using jonlib
import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
