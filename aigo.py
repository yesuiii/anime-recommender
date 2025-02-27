import gdown
import os

# Google Drive file IDs
model_file_id = "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82"  # Replace with your actual file ID
csv_file_id = "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw"  # Replace with your actual file ID

# Filenames after download
model_filename = "anime_recommender_model.pkl"
csv_filename = "users-score-2023.csv"

# Function to download files
def download_file_from_drive(file_id, output):
    if not os.path.exists(output):  # Download only if the file doesn't exist
        print(f"Downloading {output} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    else:
        print(f"{output} already exists. Skipping download.")

# Download the files
download_file_from_drive(model_file_id, model_filename)
download_file_from_drive(csv_file_id, csv_filename)

print("Files are ready!")