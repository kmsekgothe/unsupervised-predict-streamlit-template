"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from PIL import Image

# Data Loading
title_list = load_movie_titles('resources/data/new movies_df.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","EDA","Our Recommenders","Authors"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[:3000])
        movie_2 = st.selectbox('Second Option',title_list[3000:6000])
        movie_3 = st.selectbox('Third Option',title_list[6000:8862])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    bar_img = Image.open('resources/imgs/ratings.png')
    sol_img = Image.open('resources/imgs/both.png')
    solu_img = Image.open('resources/imgs/better_both.png')
    #dir_img = Image.open('resources/imgs/director.png')
    ac_img = Image.open('resources/imgs/actors.png')
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image(sol_img)
        st.write("We attempted both collaborative and content based models to compare and contrast which method \
            is the best for our mission. We decided on collaborative based filtering methods because they draw only\
                on past interaction between users and items. Furthermore, unlike their content-based counterparts, \
                    they do not require item metadata.")

        st.write("We settled on the singular value decomposition(SVD), a collaborative filtering method…not only \
            because of its good RMSE score, run time and extraordinary score on Kaggle but because of its advantage\
                of being computationally more efficient than content based methods.Profiles of users maintained in \
                    our recommender are based on the customers past interactions with items and enable powerful\
                        analysis. So we want to make sure that your users are always provided with the best \
                        possible experience. Even in industries where there isn't a large amount of competition, \
                        a good user experience will still lead to more sales and more profit")
    if page_selection == "EDA":
        st.title("Insights into the data")
        st.write("Overview of insights")
        st.image(bar_img)
        st.write("This graph compares the number of ratings per rating, but it also shows that a lot of people \
            care to rate their experience after watching a movie meaning there will not be a shortage of data ")
        st.image(ac_img)
        st.write("Actors,Directors, cast members and times change. This is why some movies can be remade and improve in ratings\
            some roles really depend on actors who are able to embody a character, and similar expressions can be said for directors \
                this app is here to keep track of all this and make the right recommendations.")
    if page_selection == "Our Recommenders":
        st.title("About Our recommenders")
        st.write("Why We Recommend These Recommenders")
        st.image(solu_img)
        st.write("Our collaborative model that takes into account the In collaborative filtering the behaviour \
            of the users is used to make recommendations to other similar users based on the preferences of other\
                users. An example of that would be recommending a product on Instagram based on a close follower or friend \
                    liking the similar product")
        st.write("This system recommends items based on similarities. it looks at the properties of an item \
            and tries to look at different items that have similar properties and recommends them to the user. \
                In Content-based Filtering, we seek to make recommendations based on how similar \
                    the properties of an item are to other items")
        st.write("Recommendation systems are forecasted to be worth 70 billion dollars in the next 3 years and a \
            recommender of your own is definitely an investment worth taking on sooner than later")
    if page_selection == "Authors":
        st.title("Starring")
        # st.write("Describe your winning approach on this page")
        image_k = Image.open('resources/imgs/karabo2.jpeg')
        #st.image(image_k)
        
        image_m = Image.open('resources/imgs/mpil2.png')
		#image_a = Image.open('resources/imgs/alyssa.jpg')
        image_a= Image.open('resources/imgs/m.jpg')
        image_h= Image.open('resources/imgs/Tshepo.jpeg')
        image_b= Image.open('resources/imgs/bohlale.jpeg')
        col1, col2 = st.columns(2)

        with col1:
            
            st.subheader("Karabo Mampuru")
            st.image(image_k,
            caption='"I can do this all day" -  Avengers: Endgame(2019)',
            width = 200)
            st.subheader("Muhammed Irfaan")
            st.image(image_a, 
            caption='"Carpe diem. Sieze the day, boys. Make your lives extraordinary" Dead Poets Society(1989)',
            width = 200)
            st.subheader("Tshepo Mokgata")
            st.image(image_h,
            caption='"May the force be with you" - Star Wars:The force awakens(2015)',
            
            width = 200)	

        with col2:
            st.subheader("Mpilenhle Hlatshwayo")
            st.image(image_m,
                caption='"I am Groot" - Guardians of the Galaxy(2014)',
                width = 200)

            
            st.subheader("Bohlale Kekana")
            st.image(image_b, 
            caption='"I would rather fight beside you than any army of thousands!\
                Let no man forget how menacing we are! We are lions" - Troy(2004)',
            width = 200)
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
