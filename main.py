import data_loading
import model
import train
#import evaluate


def main():
    print("Starting the main execution...")

    # Set the data directory
    data_directory = r'C:\Users\William\PycharmProjects\cs399\amazon-massive-dataset-1.0\1.0\data'
    print(f"Data directory: {data_directory}")

    # Load and preprocess the data
    print("Loading and preprocessing the data...")
    massive_data = data_loading.load_data(data_directory)
    if massive_data:
        train_dataset, val_dataset = data_loading.preprocess_data(massive_data)
        print(f"Datasets loaded. Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    else:
        print("No data loaded. Please check your data directory.")
        return

    # Initialize the model
    print("Initializing the model...")
    language_model = model.initialize_model()

    # Train the model
    print("Training the model...")
    if train_dataset:
        train.train_model(language_model, train_dataset)
    else:
        print("Training dataset is empty. Cannot train the model.")
        return

    # Evaluate the model
    print("Evaluating the model...")
    # if val_dataset:
    #     evaluation_results = evaluate.evaluate_model(language_model, val_dataset)
    #     print(f"Evaluation results: {evaluation_results}")
    print("Main execution finished.")

if __name__ == '__main__':
    main()
