from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction):
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}