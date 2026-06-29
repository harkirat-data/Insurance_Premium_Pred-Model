import pickle
from pathlib import Path
import pandas as pd

# Load model
MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MODEL_VERSION = "1.0.0"

# Get class labels
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([user_input])

    # Predict class
    predicted_class = model.predict(input_df)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Highest probability
    confidence = max(probabilities)

    # Create mapping {class_name: probability}
    class_probs = {
        label: round(prob, 4)
        for label, prob in zip(class_labels, probabilities)
    }

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs,
    }