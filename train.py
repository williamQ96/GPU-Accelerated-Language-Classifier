import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

def train_model(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move the model to the appropriate device
    print(f"Device: {device}")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        no_cuda=False if torch.cuda.is_available() else True,
    )

    print("Initializing the trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(AdamW(model.parameters(), lr=1e-5), None),
    )

    # Debug: for value error
    # for batch in trainer.get_train_dataloader():
    #     for k, v in batch.items():
    #         if hasattr(v, 'shape'):
    #             print(f"Key: {k}, Shape: {v.shape}, Type: {v.type()}")
    #         else:
    #             print(f"Key: {k}, Value: {v}")

    print("Starting model training...")
    trainer.train()
    print("Model training completed.")
