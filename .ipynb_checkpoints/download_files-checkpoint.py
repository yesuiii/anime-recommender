import gdown
files = {
    "anime_recommender.pkl": "1-CxYL_ePmUVw9qCkhDZVIdL1pwE-St82",
    "score.csv": "1-RiQJ2JrzxANZ1uBqiNi7WFdcjshdSZw",
}

for filename, file_id in files.items():
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)

import streamlit as st
import pandas as pd
import torch
from fastai.collab import CollabDataLoaders, collab_learner

st.write("✅ Streamlit is running!") 

@st.cache_data
def load_data():
    st.write("📥 Loading dataset...")
    chunk_size = 100000
    chunks = pd.read_csv("score.csv", chunksize=chunk_size)
    score = pd.concat(chunks)
    st.write(f"✅ Loaded {len(score)} rows")
    return score

score = load_data()

@st.cache_resource
def load_model():
    st.write("📥 Loading model...")
    dls = CollabDataLoaders.from_df(score, user_name="user_id", item_name="Anime Title", rating_name="rating", bs=512)
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.load("anime_recommender", with_opt=False, strict=False)
    st.write("✅ Model loaded successfully")
    return learn, dls

learn, dls = load_model()

st.title("🎥 Anime Recommendation System")

anime_input = st.text_input("Enter an anime title:")
if anime_input:
    st.write(f"🔍 Searching recommendations for: {anime_input}")
    st.write("⚠️ Debug: Before recommendation function")
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








    
    main()
