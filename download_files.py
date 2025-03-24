import gdown

files = {
    "anime_recommender.pkl": "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82",
    "score.csv": "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw",
}

for filename, file_id in files.items():
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

print("All files downloaded successfully!")
