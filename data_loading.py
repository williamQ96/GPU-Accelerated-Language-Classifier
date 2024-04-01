import json
import os
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer
from sklearn.model_selection import train_test_split
import torch

# Define the dataset class

# Create a mapping from locale strings to a unique integer



def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data, splitting it into training and validation sets.

    Parameters:
    - data: Dict, a dictionary with language codes as keys and loaded data as values.
    - test_size: float, the proportion of the dataset to include in the validation split.
    - random_state: int, the seed used by the random number generator for reproducibility.

    Returns:
    - train_dataset: MASSIVEDataset, the training dataset.
    - val_dataset: MASSIVEDataset, the validation dataset.
    """

    # Combine all data entries from different languages into a single list
    combined_data = [item for sublist in data.values() for item in sublist]

    # Split the combined data into training and validation sets
    train_data, val_data = train_test_split(combined_data, test_size=test_size, random_state=random_state)

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    # Create MASSIVEDataset instances for training and validation sets
    train_dataset = MASSIVEDataset(train_data, tokenizer)
    val_dataset = MASSIVEDataset(val_data, tokenizer)

    print(
        f"Preprocessing complete. Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


class MASSIVEDataset(Dataset):

    locales = ['en-US', 'de-DE', 'es-ES', 'af-ZA', 'am-ET', 'ar-SA', 'az-AZ', 'bn-BD', 'cy-GB', 'da-DK', 'el-GR',
               'fa-IR',
               'fi-FI', 'fr-FR', 'he-IL', 'hi-IN', "hu-HU", 'hy-AM', 'id-ID', 'is-IS', 'it-IT', 'ja-JP', 'jv-ID',
               'ka-GE',
               'km-KH', 'kn-IN', 'ko-KR', 'lv-LV', 'ml-IN', 'mn-MN', 'ms-MY', 'my-MM', 'nb-NO', 'nl-NL', 'pl-PL',
               'pt-PT',
               'ro-RO', 'ru-RU', 'sl-SL', 'sq-AL', 'sv-SE', 'sw-KE', 'ta-IN', 'te-IN', 'th-TH', 'tl-PH', 'tr-TR',
               'ur-PK',
               'vi-VN', 'zh-CN', 'zh-TW']
    label_to_index = {locale: index for index, locale in enumerate(locales)}

    def __init__(self, entries, tokenizer, max_length=128):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        tokens = self.tokenizer(entry['utt'],
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_tensors='pt')

        label = entry['locale']

        label_tensor = torch.tensor(self.label_to_index[label], dtype=torch.long)

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


# Function to load the data
def load_data(directory):
    print(f"Loading data from {directory}...")
    data = {}
    for file in os.listdir(directory):
        if file.endswith('.jsonl'):
            language_code = file.split('.')[0]
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data[language_code] = [json.loads(line) for line in f]
                print(f"Loaded {len(data[language_code])} entries for {language_code}")
    print("Data loading complete.")
    return data


# If this script is run as the main module, execute the following:
if __name__ == '__main__':
    # Set the data directory
    data_directory = r'C:\Users\William\PycharmProjects\cs399\amazon-massive-dataset-1.0\1.0\data'

    # Load the data
    massive_data = load_data(data_directory)

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    # Create a dataset instance for 'en-US' data
    if 'en-US' in massive_data:
        dataset = MASSIVEDataset(massive_data['en-US'], tokenizer)
        print(f"Dataset created with {len(dataset)} entries.")
    else:
        print("No 'en-US' data found.")
