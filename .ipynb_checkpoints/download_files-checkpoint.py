import streamlit as st
import pandas as pd
import torch
import os
import gdown
from fastai.collab import CollabDataLoaders, collab_learner

st.title("🎥 Anime Recommendation System")

# 🔹 Google Drive files (update IDs if needed)
FILES = {
    "anime_recommender.pkl": "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82",
    "score.csv": "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw",
}

# 🔹 Download missing files
for filename, file_id in FILES.items():
    if not os.path.exists(filename):
        st.write(f"📥 Downloading {filename}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
        st.write(f"✅ {filename} downloaded!")

# 🔹 Load dataset (optimized for large files)
@st.cache_data
def load_data():
    st.write("📥 Loading dataset...")
    chunk_size = 100000  # Adjust based on your dataset size
    chunks = pd.read_csv("score.csv", chunksize=chunk_size)
    score = pd.concat(chunks)
    st.write(f"✅ Loaded {len(score)} rows")
    return score

score = load_data()

# 🔹 Load model
@st.cache_resource
def load_model():
    st.write("📥 Loading model...")
    dls = CollabDataLoaders.from_df(score, user_name="user_id", item_name="Anime Title", rating_name="rating", bs=512)
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

    # Try loading the trained model
    try:
        learn.load("anime_recommender", with_opt=False)
        st.write("✅ Model loaded successfully")
    except Exception as e:
        st.write(f"❌ Model loading failed: {e}")
    
    return learn, dls

learn, dls = load_model()

# 🔹 Get Anime Recommendations
def get_recommendations(anime_title, top_n=5):
    anime_factors = learn.model.i_weight.weight
    if anime_title not in dls.classes['Anime Title'].items:
        return ["Anime not found in database."]
    
    idx = dls.classes['Anime Title'].o2i.get(anime_title, None)
    if idx is None:
        return ["Anime not found in database."]
    
    distances = torch.nn.functional.cosine_similarity(anime_factors, anime_factors[idx][None])
    sorted_indices = distances.argsort(descending=True)[1:top_n+1]
    
    recommendations = [dls.classes['Anime Title'].items[i] for i in sorted_indices]
    return recommendations

# 🔹 Streamlit Input
anime_input = st.text_input("Enter an anime title:")
if anime_input:
    st.write(f"🔍 Searching recommendations for: {anime_input}")
    
    recommendations = get_recommendations(anime_input)
    
    st.write("### Recommended Anime:")
    for anime in recommendations:
        st.write(f"- {anime}")
