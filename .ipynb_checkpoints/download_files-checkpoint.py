import streamlit as st
import pandas as pd
import torch
import gdown
from fastai.collab import CollabDataLoaders, collab_learner

st.set_page_config(page_title="🎥 Anime Recommendation System", layout="wide")

with st.spinner('Initializing app...'):
    st.title("🎥 Anime Recommendation System")

    FILES = {
        "score.csv": "https://drive.google.com/uc?id=1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
        "anime_recommender.pkl": "https://drive.google.com/uc?id=1-Zw0Z2MQFVBvNllFaaCGjP8j_5fM6crD",
        
    }

    for file_name, file_url in FILES.items():
        try:
            gdown.download(file_url, file_name, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {file_name}: {str(e)}")
            st.stop()

    # Load data
    try:
        score = pd.read_csv("score.csv")
    except Exception as e:
        st.error(f"Failed to load score data: {e}")
        st.stop()

@st.cache_resource
def load_model():
    st.write("📥 Loading model...")
    try:
        dls = CollabDataLoaders.from_df(
            score, 
            user_name="user_id", 
            item_name="Anime Title", 
            rating_name="rating", 
            bs=512
        )
        learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
        
        # Load model weights
        state_dict = torch.load("anime_recommender.pkl", map_location="cpu")
        learn.model.load_state_dict(state_dict)
        
        st.write("✅ Model loaded successfully!")
        return learn, dls
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

learn, dls = load_model()

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

# UI
anime_input = st.text_input("Enter an anime title:")
if anime_input:
    st.write(f"🔍 Searching recommendations for: {anime_input}")
    recommendations = get_recommendations(anime_input)
    
    st.write("### Recommended Anime:")
    for anime in recommendations:
        st.write(f"- {anime}")

# import streamlit as st
# import pandas as pd
# import torch
# import os
# import gdown
# from fastai.collab import CollabDataLoaders, collab_learner

# # Set page config first
# st.set_page_config(page_title="🎥 Anime Recommendation System", layout="wide")

# # Show loading spinner immediately
# with st.spinner('Initializing app...'):
#     st.title("🎥 Anime Recommendation System")

    # FILES = {
    #     "anime_recommender.pkl": "1-Zw0Z2MQFVBvNllFaaCGjP8j_5fM6crD",
    #     "score.csv": "1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
    # }

    # @st.cache_data(show_spinner="Downloading required files...")
    # def download_files():
    #     for filename, file_id in FILES.items():
    #         if not os.path.exists(filename):
    #             try:
    #                 gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=True)
    #             except Exception as e:
    #                 st.error(f"Failed to download {filename}: {str(e)}")
    #                 raise
    #     return True

    # try:
    #     download_files()
    # except:
    #     st.error("Failed to download required files. Please check your internet connection.")
    #     st.stop()

    # @st.cache_data(show_spinner="Loading dataset...", ttl=3600)
    # def load_data():
    #     try:
    #         score = pd.read_csv("score.csv")
    #         # Optimize memory usage
    #         score['user_id'] = score['user_id'].astype('int32')
    #         score['rating'] = score['rating'].astype('float32')
    #         return score
    #     except Exception as e:
    #         st.error(f"Failed to load dataset: {str(e)}")
    #         raise

    # try:
    #     score = load_data()
    #     st.success(f"✅ Loaded {len(score):,} rows")
    # except:
    #     st.stop()

    # @st.cache_resource(show_spinner="Loading recommendation model...")
    # def load_model():
    #     try:
    #         dls = CollabDataLoaders.from_df(
    #             score, 
    #             user_name="user_id", 
    #             item_name="Anime Title", 
    #             rating_name="rating", 
    #             bs=512
    #         )
    #         learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    #         learn.load("anime_recommender", with_opt=False)
    #         return learn, dls
    #     except Exception as e:
    #         st.error(f"Failed to load model: {str(e)}")
    #         raise

    # try:
    #     learn, dls = load_model()
    #     st.success("✅ Model loaded successfully")
    # except:
    #     st.stop()




