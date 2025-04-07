import streamlit as st
import pandas as pd
import torch
import gdown
import os
from fastai.collab import CollabDataLoaders, collab_learner
from fastai.learner import Learner
try:
    from torch.serialization import add_safe_globals
    SAFE_GLOBALS_AVAILABLE = True
except ImportError:
    st.warning("torch.serialization.add_safe_globals not available. Consider upgrading PyTorch if using Solution 2.")
    SAFE_GLOBALS_AVAILABLE = False

st.set_page_config(page_title="🎥 Anime Recommendation System", layout="wide")

@st.cache_data
def load_csv(file_name, file_url):
    if not os.path.exists(file_name):
        st.write(f"📥 Downloading {file_name}...")
        try:
            gdown.download(file_url, file_name, quiet=False)
        except Exception as e_gdown:
            st.warning(f"gdown failed for {file_name}: {e_gdown}. Trying wget...")
            try:
                os.system(f'wget --no-check-certificate "{file_url}" -O {file_name}')
                if not os.path.exists(file_name): 
                     raise FileNotFoundError(f"wget also failed to download {file_name}")
            except Exception as e_wget:
                 st.error(f"Failed to download {file_name} using gdown and wget: {e_wget}")
                 st.stop()
                 return None 

    if not os.path.exists(file_name):
         st.error(f"Required file {file_name} not found after download attempts.")
         st.stop()
         return None

    st.write(f"📄 Loading {file_name}...")
    try:
        df = pd.read_csv(file_name)
        st.success(f"✅ {file_name} loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Failed to load data from {file_name}: {e}")
        st.stop()
        return None

st.title("🎥 Anime Recommendation System")

with st.spinner('Initializing app... Checking data files...'):
    FILES = {
        "score.csv": "https://drive.google.com/file/d/1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
        "anime_recommender.pkl": "https://drive.google.com/file/d/1-a0_oGGimMSIolTZnVg7OWbv755shMxf",
    }

    score = load_csv("score.csv", FILES["score.csv"])
    if score is None:
        st.error("Could not load score data. App cannot continue.")
        st.stop()

@st.cache_resource
def load_model_and_dls(score_df, model_filename="anime_recommender.pkl", model_url=FILES["anime_recommender.pkl"]):
    st.write("🔄 Initializing DataLoaders and Model Architecture...")
    try:
        dls = CollabDataLoaders.from_df(
            score_df,
            user_name="user_id",
            item_name="Anime Title",
            rating_name="rating",
            bs=512 
        )
        learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5)) 
        st.success("✅ DataLoaders and Model Architecture ready.")
    except Exception as e:
        st.error(f"❌ Failed to initialize DataLoaders/Model: {e}")
        st.stop()
        return None, None 

    if not os.path.exists(model_filename):
        st.write(f"📥 Downloading model file ({model_filename})...")
        try:
            gdown.download(model_url, model_filename, quiet=False)
        except Exception as e_gdown:
            st.warning(f"gdown failed for {model_filename}: {e_gdown}. Trying wget...")
            try:
                os.system(f'wget --no-check-certificate "{model_url}" -O {model_filename}')
                if not os.path.exists(model_filename): 
                     raise FileNotFoundError(f"wget also failed to download {model_filename}")
            except Exception as e_wget:
                 st.error(f"Failed to download {model_filename} using gdown and wget: {e_wget}")
                 st.stop()
                 return None, None 


    st.write(f"💾 Loading model weights from {model_filename}...")
    try:
        if SAFE_GLOBALS_AVAILABLE:
             st.write("Attempting Solution 2: Load with add_safe_globals...")
             # This context manager allows controlled unpickling of specific classes
             with add_safe_globals([Learner]): # Add other necessary fastai/custom classes if needed
              
                 state_dict = torch.load(model_filename, map_location="cpu")
             learn.model.load_state_dict(state_dict)
             st.success("✅ Model state_dict loaded successfully (Solution 2 - Whitelist)!")
             return learn, dls
        else:
             st.warning("Skipping Solution 2 because add_safe_globals is not available.")
             raise RuntimeError("add_safe_globals not found, cannot use Solution 2.") # Force trying Solution 3 if needed

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.error("Please check the following:")
        st.error("1. Ensure 'anime_recommender.pkl' is correctly downloaded and not corrupted.")
        st.error("2. Verify how the model was saved (.pkl contains state_dict or full learner?).")
        st.error("3. Try uncommenting/commenting different Solutions (1, 2, 3) in the code based on how the model was saved and your PyTorch version.")
        st.error("   - Solution 1: If saved as `torch.save(learn.model.state_dict(), '...')`")
        st.error("   - Solution 2: If saved using `learn.export()` or potentially `torch.save(learn, '...')` on newer PyTorch/Fastai.")
        st.error("   - Solution 3: As a last resort if the pickle is complex and trusted.")
        st.stop()
        return None, None

# --- Actually call the function to load the model ---
learn, dls = load_model_and_dls(score)

# Check if loading was successful before proceeding
if learn is None or dls is None:
    st.error("Model and DataLoaders could not be loaded. Cannot proceed with recommendations.")
    st.stop()

def get_recommendations(input_anime_title, top_n=10): # Increased default recommendations
    try:
        # Ensure input is treated as string and handle potential non-string input gracefully
        input_anime_title = str(input_anime_title).strip()
        if not input_anime_title:
            return ["Please enter an anime title."]

        input_anime_title_lower = input_anime_title.lower()
        all_titles = list(dls.classes['Anime Title']) # Convert Index to list for easier handling
        all_titles_lower = [str(title).lower() for title in all_titles]

        if input_anime_title_lower not in all_titles_lower:
            # Suggest similar titles if not found? (Optional)
            # E.g., use fuzzywuzzy or difflib
            return [f"Anime '{input_anime_title}' not found in the database."]

        # Find the index of the input anime (case-insensitive)
        idx = all_titles_lower.index(input_anime_title_lower)

        # Get item embeddings (factors)
        anime_factors = learn.model.i_weight.weight # Accessing item embeddings

        # Calculate cosine similarity
        target_vector = anime_factors[idx].unsqueeze(0) # Keep it as a row vector [1, n_factors]
        distances = torch.nn.functional.cosine_similarity(target_vector, anime_factors) # Compare target to all others

        # Get top N similar anime indices
        # We sort descending, the first element (index 0) will be the input anime itself with similarity 1.0
        # So we take indices from 1 to top_n + 1
        sorted_indices = distances.argsort(descending=True)[1 : top_n + 1]

        # Get the corresponding anime titles
        recommended_titles = [all_titles[i] for i in sorted_indices]
        return recommended_titles

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        # You might want more specific error handling here
        return ["An error occurred while generating recommendations."]

# ======================
# USER INTERFACE
# ======================
st.markdown("---") # Separator
st.header("Get Anime Recommendations")

# Use a selectbox or autocomplete for better UX if the list isn't too large
# For now, using text input
# Add a placeholder for instruction
anime_input = st.text_input("Enter an anime title you like:", placeholder="e.g., Naruto, Death Note, Attack on Titan")

if anime_input:
    with st.spinner(f"🔍 Searching recommendations based on: {anime_input}..."):
        recommendations = get_recommendations(anime_input, top_n=10) # Get 10 recommendations

    st.write("### Recommended Anime:")
    if recommendations and not recommendations[0].startswith("Anime '") and not recommendations[0].startswith("An error") and not recommendations[0].startswith("Please enter"):
        for i, anime in enumerate(recommendations):
            st.write(f"{i+1}. {anime}")
    else:
        st.warning(recommendations[0]) # Show the message (not found, error, etc.)


# ======================
# DEBUGGING SECTION (Optional)
# ======================
with st.expander("Debug/Info Section"):
    st.write("#### Model Info:")
    if learn:
         st.write("Model Architecture Snippet:", str(learn.model)[:500] + "...") # Show only part of the model string
         # You can add more specific info like n_factors if needed
         # st.write(f"Number of factors: {learn.model.i_weight.weight.shape[1]}")
    else:
        st.write("Model not loaded.")

    st.write("#### DataLoaders Info:")
    if dls:
        st.write(f"Number of users: {dls.n_users}")
        st.write(f"Number of items (anime): {dls.n_items}")
        # Displaying classes can be long, maybe just the count or first few
        st.write(f"First 5 Anime Titles in DLS: {list(dls.classes['Anime Title'])[:5]}")
    else:
        st.write("DataLoaders not available.")

    st.write("#### Sample Score Data:")
    if score is not None:
        st.dataframe(score.head())
    else:
        st.write("Score data not loaded.")

    st.write("#### File Status:")
    for f_name in FILES.keys():
        f_path = f_name # Assuming files are in the root directory
        if os.path.exists(f_path):
             st.write(f"- `{f_name}` exists (Size: {round(os.path.getsize(f_path)/(1024*1024), 2)} MB)")
        else:
             st.write(f"- `{f_name}` does **not** exist.")

