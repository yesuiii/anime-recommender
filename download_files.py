import gdown

# Replace with your actual Google Drive file IDs
files = {
    "anime_recommender_model.pkl": "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82",
    "users-score-2023.csv": "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw",
}

for filename, file_id in files.items():
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

print("All files downloaded successfully!")
