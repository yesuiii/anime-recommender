# import streamlit as st
# import pandas as pd
# import torch
# from fastai.collab import *
# from fastai.tabular.all import *
# from fastai.collab import CollabDataLoaders
# from fastai.learner import load_learner

# score = pd.read_csv('users-score-2023.csv')

# model_path = "anime_recommender_model.pkl"
# learn = load_learner(model_path)

# def main():
#     st.title("Anime Recommendation🎌")
#     st.write("Enter your favorite anime title to get recommendations!")
#     anime_title = st.text_input("Enter an anime title:")

#     if anime_title:
#         try:
#             anime_factors = learn.model.i_weight.weight
#             idx = learn.dls.classes['Anime Title'].o2i.get(anime_title, -1)
            
#             if idx != -1:
#                 distances = nn.CosineSimilarity(dim=1)(anime_factors, anime_factors[idx][None])
#                 idxs = distances.argsort(descending=True)[1:6]
#                 recommendations = [learn.dls.classes['Anime Title'][i] for i in idxs]
                
#                 st.write("### Top 5 Recommendations:")
#                 for i, rec in enumerate(recommendations, 1):
#                     st.write(f"{i}. {rec}")
#             else:
#                 st.error("Try another anime or check the spelling.")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":




import streamlit as st
import pandas as pd
import torch
from fastai.collab import CollabDataLoaders, collab_learner

st.write("✅ Streamlit is running!")  # This should appear on the web page

@st.cache_data
def load_data():
    st.write("📥 Loading dataset...")
    score = pd.read_csv("users-score-2023.csv")
    st.write(f"✅ Loaded {len(score)} rows")
    return score

score = load_data()

@st.cache_resource
def load_model():
    st.write("📥 Loading model...")
    dls = CollabDataLoaders.from_df(score, user_name="user_id", item_name="Anime Title", rating_name="rating", bs=512)
    learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
    learn.load("anime_recommender_model", with_opt=False, strict=False)
    st.write("✅ Model loaded successfully")
    return learn, dls

learn, dls = load_model()

st.title("🎥 Anime Recommendation System")

anime_input = st.text_input("Enter an anime title:")
if anime_input:
    st.write(f"🔍 Searching recommendations for: {anime_input}")
    st.write("⚠️ Debug: Before recommendation function")

    # Recommendation logic
    def get_recommendations(anime_title, top_n=5):
        anime_factors = learn.model.i_weight.weight
        if anime_title not in dls.classes['Anime Title']:
            return ["Anime not found in database."]
        
        idx = dls.classes['Anime Title'].o2i[anime_title]
        distances = torch.nn.functional.cosine_similarity(anime_factors, anime_factors[idx][None])
        sorted_indices = distances.argsort(descending=True)[1:top_n+1]
        
        recommendations = [dls.classes['Anime Title'][i] for i in sorted_indices]
        return recommendations

    recommendations = get_recommendations(anime_input)
    st.write("⚠️ Debug: After recommendation function")
    
    st.write("### Recommended Anime:")
    for anime in recommendations:
        st.write(f"- {anime}")









    
    main()
