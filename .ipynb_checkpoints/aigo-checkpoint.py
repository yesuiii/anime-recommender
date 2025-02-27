import gdown
import os

def download_file(url, output):
    if not os.path.exists(output):  # Only download if file doesn't exist
        print(f"Downloading {output}...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"{output} already exists. Skipping download.")

model_file_id = "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82"  # Replace with your actual file ID
csv_file_id = "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw"  # Replace with your actual file ID

# File names
model_file = "anime_recommender_model.pkl"
csv_file = "users-score-2023.csv"

# Download files if not present
download_file(model_url, model_file)
download_file(csv_url, csv_file)