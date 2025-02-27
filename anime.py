import streamlit as st
import pandas as pd
import torch
from fastai.collab import *
from fastai.tabular.all import *
from fastai.collab import CollabDataLoaders
from fastai.learner import load_learner

score = pd.read_csv('users-score-2023.csv')

model_path = "anime_recommender_model.pkl"
learn = load_learner(model_path)

def main():
    st.title("Anime Recommendation🎌")
    st.write("Enter your favorite anime title to get recommendations!")
    anime_title = st.text_input("Enter an anime title:")

    if anime_title:
        try:
            anime_factors = learn.model.i_weight.weight
            idx = learn.dls.classes['Anime Title'].o2i.get(anime_title, -1)
            
            if idx != -1:
                distances = nn.CosineSimilarity(dim=1)(anime_factors, anime_factors[idx][None])
                idxs = distances.argsort(descending=True)[1:6]
                recommendations = [learn.dls.classes['Anime Title'][i] for i in idxs]
                
                st.write("### Top 5 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.error("Try another anime or check the spelling.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()