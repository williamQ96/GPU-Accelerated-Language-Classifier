import sys
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Load the model and the tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained('C:\\Users\\William\\PycharmProjects\\cs399\\results\\checkpoint-14000')
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

label_to_index = {
    'en-US': 0, 'de-DE': 1, 'es-ES': 2, 'af-ZA': 3, 'am-ET': 4, 'ar-SA': 5, 'az-AZ': 6, 'bn-BD': 7, 'cy-GB': 8,
    'da-DK': 9, 'el-GR': 10, 'fa-IR': 11, 'fi-FI': 12, 'fr-FR': 13, 'he-IL': 14, 'hi-IN': 15, "hu-HU": 16, 'hy-AM': 17,
    'id-ID': 18, 'is-IS': 19, 'it-IT': 20, 'ja-JP': 21, 'jv-ID': 22, 'ka-GE': 23, 'km-KH': 24, 'kn-IN': 25, 'ko-KR': 26,
    'lv-LV': 27, 'ml-IN': 28, 'mn-MN': 29, 'ms-MY': 30, 'my-MM': 31, 'nb-NO': 32, 'nl-NL': 33, 'pl-PL': 34,
    'pt-PT': 35, 'ro-RO': 36, 'ru-RU': 37, 'sl-SL': 38, 'sq-AL': 39, 'sv-SE': 40, 'sw-KE': 41, 'ta-IN': 42, 'te-IN': 43,
    'th-TH': 44, 'tl-PH': 45, 'tr-TR': 46,
    'ur-PK': 47, 'vi-VN': 48, 'zh-CN': 49, 'zh-TW': 50}

index_to_label = {index: locale for locale, index in label_to_index.items()}

def predict_language(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_index = logits.argmax(1).item()
        return index_to_label[predicted_index]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
        print(f"Predicted language: {predict_language(sentence)}")
    else:
        print("Please provide a sentence.")
