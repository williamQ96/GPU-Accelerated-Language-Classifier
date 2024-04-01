from transformers import XLMRobertaForSequenceClassification

def initialize_model():
    print("Initializing the model...")
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=51)
    print("Model initialized.")
    return model