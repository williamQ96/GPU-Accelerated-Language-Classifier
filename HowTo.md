# How-To Guide for GPU-Accelerated Language Classification Project

This guide will walk you through setting up and using the GPU-accelerated language classification project. 

The process involves installing necessary libraries, setting up the environment, adding data for training, updating test data, and utilizing the demo script for language prediction.

### Prerequisites

**Python 3.6 or higher**

**PyTorch** with CUDA support

**Transformers** library from Hugging Face

Access to a **CUDA-compatible GPU**


### Step 1: Install Required Libraries

Ensure Python 3 is installed on your system. 

You can verify this by running python --version or python3 --version in your command line. 

Next, install PyTorch with the appropriate CUDA version for your GPU from PyTorch's official website. 

Additionally, install the Transformers library using pip:

**pip install transformers**



### Step 2: Set the Correct Path

Locate your trained model directory, which should contain the config.json and pytorch_model.bin files from your last checkpoint, e.g., checkpoint-14000. 

Ensure that you point to this directory when loading the model in the demo.py script:


**MODEL_DIR = 'path/to/your/checkpoint/directory'**
![dir.png](screenshot%2Fdir.png)
change to your actual directory in your environment 


### Step 3: Add More Training Data (Optional)

If you wish to enhance the model's accuracy with more data:

Add new data in the JSONL format used by the Amazon Massive dataset.
Update the data_loading.py script to include new languages if necessary.
Retrain the model by running main.py.
![amazonmassive.png](screenshot%2Famazonmassive.png)


### Step 4: Update Test Data

To assess the model’s precision, you might want to add more sentences to test_data.csv. 
The format is a sentence followed by the correct language label. 
Ensure this file is in the same directory as test.py.
![data_screenshot.png](screenshot%2Fdata_screenshot.png)


### Step 5: Run the Precision Test

Execute test.py to obtain a precision score for the classification model:

python test.py

This script will process the test_data.csv file, use the model to predict languages, and compare the predictions to the correct answers, outputting a precision score.
![Weixin Screenshot_20240315234510.png](screenshot%2FWeixin%20Screenshot_20240315234510.png)


### Step 6: Use the Demo for Predictions

Use the demo.py script to predict the language of a new sentence. Run it from the command line by passing the sentence as an argument:


python3 demo.py "This is the sentence you want to classify."
The script will output the predicted language code.

![demo.png](screenshot%2Fdemo.png)
