"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""
<<<<<<< HEAD
=======
import pandas as pd
import scipy as sp # <-- The sister of Numpy, used in our code for numerical efficientcy. 
#import matplotlib.pyplot as plt
#import seaborn as sns

# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries used during sorting procedures.
import operator # <-- Convienient item retrieval during iteration 
import heapq # <-- Efficient sorting of large lists

# Imported for our sanity
import warnings
warnings.filterwarnings('ignore')

>>>>>>> da3fb67008bf99ecba0b8a43ee439018661059fc
# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import random

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

#Data Preprocessing
pred_movies = pd.merge(ratings_df,movies_df, on= 'movieId', how = 'left')
pred_movies= pred_movies.drop(['genres','rating','userId'], axis = 1)
pred_movies= pred_movies.drop_duplicates()
pred_movies = pred_movies.dropna()
movietitle = pred_movies.copy()

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
<<<<<<< HEAD
model_load_path = "resources/models/SVD.pkl"
with open(model_load_path,'rb') as file:
    model_tuned = pickle.load(file)
=======
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))
#model_2=pickle.load(open('resources/models/SVD.pkl', 'rb'))
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
>>>>>>> da3fb67008bf99ecba0b8a43ee439018661059fc

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.
    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.
    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.
    """
<<<<<<< HEAD
    #Data preprosessing
=======
    
>>>>>>> da3fb67008bf99ecba0b8a43ee439018661059fc
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    a_train = load_df.build_full_trainset()
    print(a_train)
    predictions = []
    for ui in a_train.all_users():
<<<<<<< HEAD
        predictions.append(model_tuned.predict(iid=item_id, uid=ui, verbose=False))
=======
        predictions.append(model.predict(item_id,ui))
>>>>>>> da3fb67008bf99ecba0b8a43ee439018661059fc
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.
    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.
    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.
    """
    # Store the id of users
    id_store = []
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        # movie_id = movietitle[movietitle['title'] == i]['movieId'].values[0]
        predictions = prediction_item(item_id=i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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
    
    book_ratings = pd.read_csv('resources/data/ratings.csv')
    book_ratings.drop(['timestamp'], axis=1,inplace=True)
    book_ratings = book_ratings[:4000]
    print(book_ratings.head())

    
    
    print(movies_df.head())
    movies_df['year'] = movies_df['title'].str.extract(r"\((\d+)\)", expand=False)
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    
    print("newnewnewnwneenwenenwenenwenwenwenenwenwenwenwenwenwenwenwenwenwe")
    userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
           
         ] 
    inputMovies = pd.DataFrame(userInput)
    print(inputMovies.head(3))
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    
    print(inputId.head(4))
    inputMovies = pd.merge(inputId, inputMovies)
    print(inputMovies.head(4))
    
    inputMovies = inputMovies.drop('year', 1)
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    print(userSubset.head(9))

    choice = max(set(list(userSubset['userId'])), key = list(userSubset['userId']).count) 
    print("444444444444444444444444444444444444444444444444444444444")
    print(choice)
    util_matrix = book_ratings.pivot_table(index=['userId'],
                                       columns=['movieId'],
                                       values='rating')
    print("1111111111111111111111")
    # Normalize each row (a given user's ratings) of the utility matrix
    util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    # Fill Nan values with 0's, transpose matrix, and drop users with no ratings
    util_matrix_norm.fillna(0, inplace=True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    # Save the utility matrix in scipy's sparse matrix format
    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)
    print("22222222222222222222222222")
    # Compute the similarity matrix using the cosine similarity metric
    user_similarity = cosine_similarity(util_matrix_sparse.T)
    # Save the matrix as a dataframe to allow for easier indexing  
    user_sim_df = pd.DataFrame(user_similarity,
                            index = util_matrix_norm.columns,
                            columns = util_matrix_norm.columns)
    print("33333333333333333333333333333333333333333333")
    
    #below method uses 1 id to recommend, so maybe if we have 3 movies that can spit out 1 ID this will work
     
    def collab_generate_top_N_recommendations(user, N=top_n, k=20):
    # Cold-start problem - no ratings given by the reference user. 
    # With no further user data, we solve this by simply recommending
    # the top-N most popular books in the item catalog. 
        if user not in user_sim_df.columns:
            return book_ratings.groupby('title').mean().sort_values(by='rating',
                                            ascending=False).index[:N].to_list()

        # Gather the k users which are most similar to the reference user 
        sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:k+1]
        favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
        most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users

        for i in sim_users:
            # Maximum rating given by the current user to an item 
            max_score = util_matrix_norm.loc[:, i].max()
            # Save the names of items maximally rated by the current user   
            favorite_user_items.append(util_matrix_norm[util_matrix_norm.loc[:, i]==max_score].index.tolist())

        # Loop over each user's favorite items and tally which ones are 
        # most popular overall.
        for item_collection in range(len(favorite_user_items)):
            for item in favorite_user_items[item_collection]:
                if item in most_common_favorites:
                    most_common_favorites[item] += 1
                else:
                    most_common_favorites[item] = 1
        # Sort the overall most popular items and return the top-N instances
        sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:N]
        top_N = [x[0] for x in sorted_list]
        return top_N
    nums = collab_generate_top_N_recommendations(20)
    print(nums)
    recommended = []    
    print(movies_df.head(5))
    print("----------------------------------------")
    
    for i in nums:
        temp = []
        new_df = movies_df[['title']][movies_df['movieId']==i]
    
        temp = list(new_df['title'])
        recommended.append(temp[0])
    print(list(recommended))

    return recommended
    
    # print(movies_df.head())
    # movies_df['year'] = movies_df['title'].str.extract(r"\((\d+)\)", expand=False)
    # movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    # print(movies_df['year'].head(4))
    # movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    # movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    # movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    # movies = movies_df.drop('genres', 1)
    
    # userInput = [
    #         {'title':'Breakfast Club, The', 'rating':5},
    #         {'title':'Toy Story', 'rating':3.5},
    #         {'title':'Jumanji', 'rating':2},
    #      ] 
    # inputMovies = pd.DataFrame(userInput)
    
    # inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    # inputMovies = pd.merge(inputId, inputMovies)
    # inputMovies = inputMovies.drop('year', 1)
    # print(inputMovies.head(5))
    
    # userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    # print("user subset")
    # print(userSubset.head())
    # userSubsetGroup = userSubset.groupby(['userId'])
    # #print(userSubsetGroup.head())
    # userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
    # #userSubsetGroup[0][1]
    # userSubsetGroup = userSubsetGroup[0:100]
    # users=[]
    # for name, group in userSubsetGroup:
        
    #     #Let's start by sorting the input and current user group so the values aren't mixed up later on
    #     users.append(name)
    #     group = group.sort_values(by='movieId')
    #     inputMovies = inputMovies.sort_values(by='movieId')
        
    #     #Get the N (total similar movies watched) for the formula 
    #     nRatings = len(group)
        
    #     #Get the review scores for the movies that they both have in common
    #     temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        
    #     ###For Debugging Purpose
    #     if nRatings<5:
    #         print(inputMovies['movieId'].isin(group['movieId'].tolist()))
    #     #    break
    #     #else:
    #     #    continue
        
    #     #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    #     tempRatingList = temp_df['rating'].tolist()
        
    #     #Let's also put the current user group reviews in a list format
    #     tempGroupList = group['rating'].tolist()
    # print("after long pears")
    # users = users[0:10]
    # print(users)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # #movies_df
    # indices = pd.Series(movies_df['title'])
    # #movie_ids = pred_movies(movie_list)
    # movie_ids = users
    # print("11111111111111111111111")
    # df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    # for i in movie_ids :
    #     df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
    # # Getting the cosine similarity matrix
    # print(df_init_users.head(4))
    # print("222222222222222222222222222")
    # cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    # idx_1 = indices[indices == movie_list[0]].index[0]
    # idx_2 = indices[indices == movie_list[1]].index[0]
    # idx_3 = indices[indices == movie_list[2]].index[0]
    # # Creating a Series with the similarity scores in descending order
    # print("33333333333333333333333333333")
    # rank_1 = cosine_sim[idx_1]
    # rank_2 = cosine_sim[idx_2]
    # rank_3 = cosine_sim[idx_3]
    # # Calculating the scores
    # score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    # score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    # score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    #  # Appending the names of movies
    # listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)
    # recommended_movies = []
    # # Choose top 50
    # print("444444444444444444444444444444444444444444")
    # top_50_indexes = list(listings.iloc[1:50].index)
    # # Removing chosen movies
    # top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # for i in top_indexes[:top_n]:
    #     recommended_movies.append(list(movies_df['title'])[i])
        
    # #return recommended_movies
    
    # return ["title 1","title 2","title 3","title 4","title 5","title 6","title 7","title 8","title 9","title 10"]

    # book_ratings = pd.read_csv('resources/data/ratings.csv')
    #     book_ratings.drop(['timestamp'], axis=1,inplace=True)
    #     book_ratings = book_ratings[:4000]
    #     print(book_ratings.head())

    #     print (f'Number of ratings in dataset: {book_ratings.shape[0]}')

    #     util_matrix = book_ratings.pivot_table(index=['userId'],
    #                                     columns=['movieId'],
    #                                     values='rating')
    #     print("1111111111111111111111")
    #     # Normalize each row (a given user's ratings) of the utility matrix
    #     util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    #     # Fill Nan values with 0's, transpose matrix, and drop users with no ratings
    #     util_matrix_norm.fillna(0, inplace=True)
    #     util_matrix_norm = util_matrix_norm.T
    #     util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    #     # Save the utility matrix in scipy's sparse matrix format
    #     util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)
    #     print("22222222222222222222222222")
    #     # Compute the similarity matrix using the cosine similarity metric
    #     user_similarity = cosine_similarity(util_matrix_sparse.T)
    #     # Save the matrix as a dataframe to allow for easier indexing  
    #     user_sim_df = pd.DataFrame(user_similarity,
    #                             index = util_matrix_norm.columns,
    #                             columns = util_matrix_norm.columns)
    #     print("33333333333333333333333333333333333333333333")
    #     def collab_generate_top_N_recommendations(user, N=10, k=20):
    #     # Cold-start problem - no ratings given by the reference user. 
    #     # With no further user data, we solve this by simply recommending
    #     # the top-N most popular books in the item catalog. 
    #         if user not in user_sim_df.columns:
    #             return book_ratings.groupby('title').mean().sort_values(by='rating',
    #                                             ascending=False).index[:N].to_list()

    #         # Gather the k users which are most similar to the reference user 
    #         sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:k+1]
    #         favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
    #         most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users

    #         for i in sim_users:
    #             # Maximum rating given by the current user to an item 
    #             max_score = util_matrix_norm.loc[:, i].max()
    #             # Save the names of items maximally rated by the current user   
    #             favorite_user_items.append(util_matrix_norm[util_matrix_norm.loc[:, i]==max_score].index.tolist())

<<<<<<< HEAD
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId'] == movie_ids[0]]
    for i in movie_ids[1:]:
        df_init_users = df_init_users.append(ratings_df[ratings_df['userId'] == i])

    # Getting the user-item matrix
    df_init_users = pd.merge(df_init_users, movietitle, on='movieId', how='left')
    df_init_users = df_init_users.dropna()
    users_matrix = df_init_users.groupby(['title', 'userId'])['rating'].max().unstack()
    for i in movie_list:
        if i not in users_matrix.index.values.tolist():
            df_nan = pd.DataFrame([[(np.NaN)] * len(users_matrix.columns)], index=[i], columns=users_matrix.columns)
            users_matrix = users_matrix.append(df_nan)

    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(users_matrix.fillna(0))
    indices = pd.Series(users_matrix.index)
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]

    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending=False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending=False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending=False)

    # Appending the names of movies
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending=False)
    recommended_movies = []

    # Choose top 50
    top_50_indx = list(listings.iloc[1:50].index)

    # Removing chosen movies
    top_indx = np.setdiff1d(top_50_indx, [idx_1, idx_2, idx_3])
    random.shuffle(top_indx)
    for j in top_indx[:top_n]:
        recommended_movies.append(indices[j])

    return recommended_movies
=======
    #         # Loop over each user's favorite items and tally which ones are 
    #         # most popular overall.
    #         for item_collection in range(len(favorite_user_items)):
    #             for item in favorite_user_items[item_collection]:
    #                 if item in most_common_favorites:
    #                     most_common_favorites[item] += 1
    #                 else:
    #                     most_common_favorites[item] = 1
    #         # Sort the overall most popular items and return the top-N instances
    #         sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:N]
    #         top_N = [x[0] for x in sorted_list]
    #         return top_N
    #     nums = collab_generate_top_N_recommendations(20)
    #     print(nums)
    #     recommended = ["t"]    
    #     print(movies_df.head(5))
    #     for i in nums:
    #         recommended.append(movies_df['title'][movies_df['movieId']==i])
    #     print(list(recommended))
>>>>>>> da3fb67008bf99ecba0b8a43ee439018661059fc
