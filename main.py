import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import json

# --- SENTIMENT ANALYSIS SETUP ---

@st.cache_resource
def load_sentiment_model_and_data():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    model = load_model('simple_rnn_imdb.h5')
    return model, word_index, reverse_word_index

model, word_index, reverse_word_index = load_sentiment_model_and_data()

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- MOVIE RECOMMENDATION SETUP ---

@st.cache_data
def load_and_merge_data(movies_path, credits_path):
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    # FIX: The 'title' column exists in both DataFrames. To prevent pandas from
    # creating 'title_x' and 'title_y' columns during the merge, we drop the
    # redundant 'title' column from credits_df.
    credits_df = credits_df.drop('title', axis=1)
    
    # Merge the two dataframes on the 'id' column.
    movies_df = movies_df.merge(credits_df.rename(columns={'movie_id': 'id'}), on='id')
    
    def parse_genres(genres_str):
        try:
            # Safely parse the JSON string in the 'genres' column
            if isinstance(genres_str, str):
                 return [i['name'] for i in json.loads(genres_str)]
            return []
        except (json.JSONDecodeError, TypeError):
            return []
    
    movies_df['genre_names'] = movies_df['genres'].apply(parse_genres)
    return movies_df

def recommend_movies(df, genre, top_n=5):
    genre_df = df[df['genre_names'].apply(lambda x: genre in x)]
    qualified = genre_df.copy()

    # Filter out movies with too few votes.
    if 'vote_count' in qualified.columns:
        qualified = qualified[qualified['vote_count'] >= 100]
    
    # Sort by vote average (higher is better).
    if 'vote_average' in qualified.columns:
        qualified = qualified.sort_values('vote_average', ascending=False)
    
    return qualified.head(top_n)

# --- Load Data ---
try:
    movies_df = load_and_merge_data('tmdb_5000_movies.csv', 'tmdb_5000_credits.csv')
    all_genres = sorted(list(set([genre for sublist in movies_df['genre_names'] for genre in sublist])))
except FileNotFoundError:
    movies_df = None
    all_genres = []

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")

# Part 1: Sentiment Analysis
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')
user_input = st.text_area('Movie Review', key="sentiment_input")

if st.button('Classify Sentiment'):
    if user_input:
        with st.spinner('Analyzing...'):
            processed_input = preprocess_text(user_input)
            prediction = model.predict(processed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            
            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
    else:
        st.warning('Please enter a movie review first.')

st.markdown("---")

# Part 2: Movie Recommendation
st.title('🍿 Movie Recommender')
st.write('Select a genre to get movie recommendations.')

if movies_df is not None:
    selected_genre = st.selectbox('Choose a genre:', all_genres)

    if st.button('Get Recommendations'):
        if selected_genre:
            with st.spinner(f'Finding the best movies in {selected_genre}...'):
                st.subheader(f'Top 5 Recommended Movies in "{selected_genre}"')
                recommendations = recommend_movies(movies_df, selected_genre)

                if not recommendations.empty:
                    # Display recommendations in columns
                    cols = st.columns(len(recommendations))
                    for i, (index, row) in enumerate(recommendations.iterrows()):
                        with cols[i]:
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"⭐ **{row['vote_average']}** / 10")
                            with st.expander("See overview"):
                                st.write(row['overview'])
                else:
                    st.warning(f'No movies found for "{selected_genre}" with at least 100 votes.')
else:
    st.error("`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` not found. Please ensure both files are in the same directory.")
