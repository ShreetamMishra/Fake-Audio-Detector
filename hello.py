import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile
import io

# Set your Kaggle credentials directly
os.environ['KAGGLE_USERNAME'] = "shreetammishra"
os.environ['KAGGLE_KEY'] = "51e1ed72c961034875cb6fb6531bd1d2"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify the dataset and file
dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'  # Dataset path
file_name = 'for-2sec/training/fake'  # Example file within the dataset

# Download the dataset file
api.dataset_download_file(dataset, file_name)

# Unzipping the file content
with zipfile.ZipFile(file_name + '.zip', 'r') as zip_ref:
    zip_ref.extractall(".")

# Load the file content into a Pandas DataFrame
df = pd.read_csv(file_name)
print(df.head())
