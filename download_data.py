import os
import subprocess

def download_dataset():
    # Ensure your Kaggle API token environment variable is set
    # e.g., os.environ['KAGGLE_API_TOKEN'] = "YOUR_TOKEN"
    if 'KAGGLE_API_TOKEN' not in os.environ:
        print("Warning: KAGGLE_API_TOKEN environment variable not set.")
    
    print("Downloading dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "reubensuju/celeb-df-v2"], check=True)
    
    print("Unzipping dataset...")
    subprocess.run(["unzip", "-q", "-n", "celeb-df-v2.zip", "-d", "deepfake_dataset/"], check=True)
    print("Dataset ready in 'deepfake_dataset/' folder.")

if __name__ == "__main__":
    download_dataset()
