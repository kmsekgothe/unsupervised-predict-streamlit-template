"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
# Importing data
#/home/explore-student/unsupervised_data/unsupervised_movie_data/
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
path = "/home/explore-student/unsupervised_data/unsupervised_movie_data";

movies_df = pd.read_csv('resources/data/movies.csv',sep=',')
imdb_data = pd.read_csv('resources/data/imdb_data.csv')
#cv = joblib.load(open('resources/models/Vectorizer2.pkl', 'rb'))

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    
    
    df = imdb_data[['movieId','title_cast','director', 'plot_keywords']]
    df = df.merge(movies[['movieId', 'genres', 'title']], on='movieId', how='inner')
    df['year'] = df['title'].str.extract(r"\((\d+)\)", expand=False)
    
    df['title_cast'] = df.title_cast.astype(str)
    
    df['plot_keywords'] = df.plot_keywords.astype(str)
    df['genres'] = df.genres.astype(str)
    df['director'] = df.director.astype(str)

   
    df['director'] = df['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
    df['title_cast'] = df['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))

    df['title_cast'] = df['title_cast'].map(lambda x: x.split('|')[:3])

    df['plot_keywords'] = df['plot_keywords'].map(lambda x: x.split('|')[:5])
    df['plot_keywords'] = df['plot_keywords'].apply(lambda x: " ".join(x))

    df['genres'] = df['genres'].map(lambda x: x.lower().split('|'))
    df['genres'] = df['genres'].apply(lambda x: " ".join(x))
    
    df['corpus'] = ''
    corpus = []
    
    columns = [ 'plot_keywords', 'genres', 'year']
    for i in range(0, len(df['movieId'])):
        words = ''
        for col in columns:
            words = words + str(df.iloc[i][col]) + " "        
        corpus.append(words)

    df['corpus'] = corpus
    df.drop(columns=['title_cast', 'director', 'plot_keywords', 'genres', 'year'], inplace=True)
   
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    
    movies_subset = movies[:subset_size]
    
    the_subset = df[:subset_size]
    
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    count_vec = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    count_matrix = count_vec.fit_transform(data['keyWords'])
    
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
