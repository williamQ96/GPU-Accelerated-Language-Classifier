import csv
import sys
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Path to the trained model directory
MODEL_DIR = 'C:\\Users\\William\\PycharmProjects\\cs399\\results\\checkpoint-14000'  # Replace with your model directory

# Load the model and tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the appropriate device


# Function to predict the language of a sentence
def predict_language(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_index = logits.argmax(1).item()
    return predicted_index


# Load the test data from a CSV file and predict
def test_model_from_csv(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        correct_predictions = 0
        total_predictions = 0

        for row in reader:
            sentence, true_label = row
            true_label = int(true_label)
            predicted_label = predict_language(sentence)

            if predicted_label == true_label:
                correct_predictions += 1
            total_predictions += 1

    success_rate = correct_predictions / total_predictions
    return success_rate


# Provide the path to your CSV file here
csv_file_path = 'C:\\Users\\William\\PycharmProjects\\cs399\\test_data.csv'  # Replace with your CSV file path

# Run the test and print the success rate
if __name__ == "__main__":
    success_rate = test_model_from_csv(csv_file_path)
    print(f"Success rate: {success_rate:.2%}")