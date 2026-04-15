# predict.py

import numpy as np
from src.utils import load_model
import src.config as config

# 1. Load trained model
model = load_model(config.MODEL_PATH)

# 2. Example input (MUST match dataset columns order)
# age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal

sample = np.array([[
    63, 1, 3, 145, 233, 1, 0,
    150, 0, 2.3, 0, 0, 1
]])

# 3. Predict
prediction = model.predict(sample)

# 4. Output
print("Prediction (0 = no disease, 1 = disease):", prediction)
