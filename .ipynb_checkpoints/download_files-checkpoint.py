# import streamlit as st
# import pandas as pd
# import torch
# import gdown
# from fastai.collab import CollabDataLoaders, collab_learner

# st.set_page_config(page_title="🎥 Anime Recommendation System", layout="wide")

# with st.spinner('Initializing app...'):
#     st.title("🎥 Anime Recommendation System")

#     FILES = {
#         "score.csv": "https://drive.google.com/uc?id=1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
#         "anime_recommender.pkl": "https://drive.google.com/uc?id=1-Zw0Z2MQFVBvNllFaaCGjP8j_5fM6crD",
#     }

#     # Download files
#     for file_name, file_url in FILES.items():
#         try:
#             gdown.download(file_url, file_name, quiet=False)
#         except Exception as e:
#             st.error(f"Failed to download {file_name}: {str(e)}")
#             st.stop()

#     # Load data
#     try:
#         score = pd.read_csv("score.csv")
#     except Exception as e:
#         st.error(f"Failed to load score data: {e}")
#         st.stop()

# from fastai.learner import Learner
# from torch.serialization import add_safe_globals

# @st.cache_resource
# def load_model():
#     try:
#         dls = CollabDataLoaders.from_df(score, user_name="user_id", item_name="Anime Title", rating_name="rating", bs=512)
#         learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
        
#         # ✅ Allow Fastai's Learner class
#         with add_safe_globals([Learner]):
#             state_dict = torch.load("anime_recommender.pkl", map_location="cpu")
#         learn.model.load_state_dict(state_dict)
        
#         st.success("✅ Model loaded successfully!")
#         return learn, dls
#     except Exception as e:
#         st.error(f"❌ Model loading failed: {e}")
#         st.stop()

# def get_recommendations(anime_title, top_n=5):
#     try:
#         anime_title_lower = anime_title.lower()
#         all_titles = dls.classes['Anime Title'].items
#         all_titles_lower = [title.lower() for title in all_titles]
        
#         if anime_title_lower not in all_titles_lower:
#             return ["Anime not found in database."]
        
#         idx = all_titles_lower.index(anime_title_lower)
#         anime_factors = learn.model.i_weight.weight
#         distances = torch.nn.functional.cosine_similarity(
#             anime_factors, 
#             anime_factors[idx][None]
#         )
#         sorted_indices = distances.argsort(descending=True)[1:top_n+1]
        
#         return [all_titles[i] for i in sorted_indices]
#     except Exception as e:
#         return [f"Error: {str(e)}"]

# # UI
# anime_input = st.text_input("Enter an anime title:")
# if anime_input:
#     st.write(f"🔍 Searching recommendations for: {anime_input}")
#     recommendations = get_recommendations(anime_input)
    
#     st.write("### Recommended Anime:")
#     for anime in recommendations:
#         st.write(f"- {anime}")



import streamlit as st
import pandas as pd
import torch
import gdown
import os
from fastai.collab import CollabDataLoaders, collab_learner
from fastai.learner import Learner
from torch.serialization import add_safe_globals

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(page_title="🎥 Anime Recommendation System", layout="wide")

# ======================
# FILE DOWNLOAD
# ======================
with st.spinner('Initializing app...'):
    st.title("🎥 Anime Recommendation System")

    FILES = {
        "score.csv": "https://drive.google.com/uc?id=1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
        "anime_recommender.pkl": "https://drive.google.com/uc?id=1-Zw0Z2MQFVBvNllFaaCGjP8j_5fM6crD",
    }

    # Download files with fallback to wget if gdown fails
    for file_name, file_url in FILES.items():
        if not os.path.exists(file_name):
            try:
                gdown.download(file_url, file_name, quiet=False)
            except:
                st.warning(f"gdown failed for {file_name}, trying wget...")
                os.system(f'wget --no-check-certificate "{file_url}" -O {file_name}')

    # Verify downloads
    if not all(os.path.exists(f) for f in FILES.keys()):
        st.error("Failed to download required files")
        st.stop()

    # Load data
    try:
        score = pd.read_csv("score.csv")
    except Exception as e:
        st.error(f"Failed to load score data: {e}")
        st.stop()

# ======================
# MODEL LOADING (3 SOLUTIONS)
# ======================
@st.cache_resource
def load_model():
    st.write("📥 Loading model...")
    try:
        # Initialize model architecture
        dls = CollabDataLoaders.from_df(
            score, 
            user_name="user_id", 
            item_name="Anime Title", 
            rating_name="rating", 
            bs=512
        )
        learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

        # =============================================
        # SOLUTION 1: Recommended - Load weights only
        # (Requires the .pkl to contain just state_dict)
        # =============================================
        # learn.model.load_state_dict(torch.load("anime_recommender.pkl"))
        
        # =============================================
        # SOLUTION 2: Whitelist Fastai components
        # (For PyTorch 2.6+ with security)
        # =============================================
        with add_safe_globals([Learner]):
            state_dict = torch.load("anime_recommender.pkl", map_location="cpu")
        learn.model.load_state_dict(state_dict)
        
    #     # =============================================
    #     # SOLUTION 3: Unsafe load (only for trusted files)
    #     # =============================================
    #     # state_dict = torch.load("anime_recommender.pkl", 
    #     #                        map_location="cpu", 
    #     #                        weights_only=False)
    #     # learn.model.load_state_dict(state_dict)

    #     st.success("✅ Model loaded successfully!")
    #     return learn, dls
        
    # except Exception as e:
    #     st.error(f"❌ Model loading failed: {e}")
    #     st.error("Try solutions in this order:")
    #     st.error("1. Re-save the model as state_dict only")
    #     st.error("2. Use the whitelist approach (current solution)")
    #     st.error("3. As last resort, use weights_only=False")
    #     st.stop()

# ======================
# RECOMMENDATION ENGINE
# ======================
def get_recommendations(anime_title, top_n=5):
    try:
        anime_title_lower = anime_title.lower()
        all_titles = dls.classes['Anime Title'].items
        all_titles_lower = [title.lower() for title in all_titles]
        
        if anime_title_lower not in all_titles_lower:
            return ["Anime not found in database."]
        
        idx = all_titles_lower.index(anime_title_lower)
        anime_factors = learn.model.i_weight.weight
        distances = torch.nn.functional.cosine_similarity(
            anime_factors, 
            anime_factors[idx][None]
        )
        sorted_indices = distances.argsort(descending=True)[1:top_n+1]
        
        return [all_titles[i] for i in sorted_indices]
    except Exception as e:
        return [f"Error: {str(e)}"]

# ======================
# USER INTERFACE
# ======================
learn, dls = load_model()

anime_input = st.text_input("Enter an anime title:")
if anime_input:
    st.write(f"🔍 Searching recommendations for: {anime_input}")
    recommendations = get_recommendations(anime_input)
    
    st.write("### Recommended Anime:")
    for anime in recommendations:
        st.write(f"- {anime}")

# ======================
# DEBUGGING SECTION
# ======================
with st.expander("Debug Info"):
    st.write("Model architecture:", learn.model)
    st.write("Sample data:", score.head())
    if os.path.exists("anime_recommender.pkl"):
        st.write("Model file exists (size: {} MB)".format(
            round(os.path.getsize("anime_recommender.pkl")/(1024*1024), 2)))

