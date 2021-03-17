from flask import Flask,render_template
import pyrebase
# import firebase_admin


from datetime import datetime

import config # to hide TMDB API keys
import requests # to make TMDB API calls
import locale # to format currency as USD
locale.setlocale( locale.LC_ALL, '' )

import pandas as pd
from datetime import datetime
import time
import config  # to hide TMDB API keys
import requests  # to make TMDB API calls
import locale  # to format currency as USD

locale.setlocale(locale.LC_ALL, '')
import pandas as pd
from rake_nltk import Rake
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk as nltk
import zipfile
from pip._vendor.distlib import metadata
row_name=[]
Recommendation=[]
api_key = 'f86e455366e5b5d9b387ba94241e2c9c'
action = 28
adventure = 12
comedy = 35
drama = 18
horror = 27

Dataset = "tmdb-movies-dataset"
# Provide downloaded zip file location
with zipfile.ZipFile("tmdb-movies-dataset.zip", "r") as z:
    z.extractall(".")
allData = pd.read_csv("tmdb_movies_data.csv")  # Provide downloaded csv file location(csv file is inside zip)
# we get 1000 movies from database to shorten time elapsed to retrieve data
allData_df = allData.head(1000)
# allData_df = allData
copy_df = allData.copy()

copy_df.drop(
    columns=['keywords','imdb_id', 'popularity', 'budget', 'revenue', 'homepage',
             'tagline', 'runtime', 'production_companies', 'vote_count', 'budget_adj','revenue_adj'],
    inplace=True)

allData_df.drop(
    columns=['id', 'keywords', 'release_year', 'release_date', 'imdb_id', 'popularity', 'budget', 'revenue', 'homepage',
             'tagline', 'runtime', 'production_companies', 'vote_count', 'budget_adj', 'vote_average', 'revenue_adj'],
    inplace=True)

allData_df.rename(columns={'original_title': 'movie', 'cast': 'actors',
                           'overview': 'description', 'genres': 'categories'}, inplace=True)
print(allData_df)
print("Copy df")
print(copy_df)
print("--categories---")
copy_df['genres'] = copy_df['genres'].map(lambda x: str(x).split('|'))
print(copy_df['genres'])
copy_df['cast'] = copy_df['cast'].map(lambda x: str(x).split('|'))
copy_df['director'] = copy_df['director'].map(lambda x: str(x).lower().split('|'))

# Getting images for movies
def getBackgroundImage(id):
    try:
        response = requests.get('https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=' + api_key + '&language=en-US')
        rJson = response.json()
        value = rJson["backdrop_path"]
    except KeyError as e:
        value = ""      
    return value
    
    
def getPosterImage(id):
    try:
        response = requests.get('https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=' + api_key + '&language=en-US')
        rJson = response.json()
        value = rJson["poster_path"]
    except KeyError as e:
        value = ""      
    return value
    
# Getting 10 random action, adventure, comedy, drama, horror
def homepage_movies(genre):
    homepage_movies = []
    counter = 0
    for index, row in copy_df.iterrows():
        row['genres'] = row['genres'][0]
        # print(row['genres'])
        if row['genres'] == genre:
            if counter < 10:
                back_image = getBackgroundImage(row['id'])
                post_image = getPosterImage(row['id'])
                if (back_image == "" or post_image ==""):
                    continue
                else :
                    homepage_dict = {'title': row['original_title'], 'id': row['id'], 'release_date': row['release_date'], 'vote_average': row['vote_average'], 'overview':row['overview'], 'poster_path':post_image, 'backdrop_path':back_image, 'actors':row['cast'], 'directors':row['director']}
                    homepage_movies.append(homepage_dict)
                    counter = counter + 1
    return homepage_movies

def homepage_recommendations(movieNames):
    homepage_recommendations = []
    counter = 0
    for movie in movieNames:
        for index, row in copy_df.iterrows():
            # print(row['genres'])
            if row['original_title'] == movie:
                if counter < 10:
                    back_image = getBackgroundImage(row['id'])
                    post_image = getPosterImage(row['id'])
                    if (back_image == "" or post_image ==""):
                        continue
                    else :
                        recommended_dict = {'title': row['original_title'], 'id': row['id'], 'release_date': row['release_date'], 'vote_average': row['vote_average'], 'overview':row['overview'], 'poster_path':post_image, 'backdrop_path':back_image, 'actors':row['cast'], 'directors':row['director']}
                        homepage_recommendations.append(recommended_dict)
                        counter = counter + 1
    return homepage_recommendations
            
print(homepage_movies("Action"))
        
# def popular_movies:
    
    


# Data preparation for tf-idf and cosine similarity algorithms
allData_df['actors'] = allData_df['actors'].map(lambda x: str(x).split('|'))

# putting the genres in a list of words
allData_df['categories'] = allData_df['categories'].map(lambda x: str(x).lower().split('|'))

# allData_df['director'] = allData_df['director'].map(lambda x: str(x).split(' '))
allData_df['director'] = allData_df['director'].map(lambda x: str(x).lower().split('|'))
print("--directors")
print(allData_df['director'])
# merging together first and last name for each actor and director, so it's considered as one word
# and there is no mix up between people sharing a first name
for index, row in allData_df.iterrows():
    row['actors'] = [x.lower().replace(' ', '') for x in row['actors']]
    row['director'] = [x.replace(' ', '') for x in row['director']]
    # row['director'] = row['director'].replace(' ','')
    # row['director'] = ''.join(row['director']).lower()
# initializing the new column
allData_df['key_words'] = ""

print("\n\n\nAFTER MODIFYING ACTORS DIRECTOR AND CATEGORIES")
print(allData_df['director'])
print(allData_df)

# allData_df['key_words'] = ""

for index, row in allData_df.iterrows():
    description = row['description']

    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all punctuation characters as well
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(str(description))

    # getting the dictionary with key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    # assigning the key words to the new column for the corresponding movie
    row['key_words'] = list(key_words_dict_scores.keys())

print(allData_df)

# dropping the Plot column
allData_df.drop(columns=['description'], inplace=True)
allData_df.set_index('movie', inplace=True)

print("AFTER CREATING KEYWORDS - CONSISTING OF WORDS FROM description")
print(allData_df)

allData_df['bag_of_words'] = ''
columns = allData_df.columns
for index, row in allData_df.iterrows():
    words = ''
    for col in columns:
        if col != 'movie':
            words = words + ' '.join(row[col]) + ' '
        else:
            words = words + row[col] + ' '
    row['bag_of_words'] = words

allData_df.drop(columns=[col for col in allData_df.columns if col != 'bag_of_words' and col != 'movie'], inplace=True)

print("---------New data list AFTER COMBINING ALL WORDS except TITLE -------------")
print(allData_df)
print(allData_df['bag_of_words'])

# getting unique words from bag_of_words
uniqueWordsNew = list(allData_df['bag_of_words'].str.split(' ', expand=True).stack().unique())
# print(uniqueWords)
uniqueWordsNew.remove('')
print("-new unique word")
# print(uniqueWordsNew)

numOfWords = []
totalWords = []
tfWords = []
tfIdfWords = []

k = 0

for index, row in allData_df.iterrows():
    # print(row['bag_of_words'])
    totalWords.append(row['bag_of_words'])

print("total documents")
print(totalWords[0])
print("----------------------------")
print(totalWords[2])

# perform tf-idf operations using tfidfvectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(totalWords)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
tfidf_df = pd.DataFrame(denselist, columns=feature_names)

# cosine_sim = linear_kernel(tfidf_df, tfidf_df)
# create cosine similarity matrix by using tf_idf dataset
cosine_similarity = cosine_similarity(tfidf_df, tfidf_df)
print("--Cosine sim--")
print(cosine_similarity)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
print("--asdasdsadas-----")
print(allData_df.index)
indices = pd.Series(allData_df.index)
print("--lel-----")
print(indices)
print("----------------------------------------------------------------------------------------")
movieList = []


# function to recommend movies that are similar to movies in movieList
def recommendations(movieList, cosine_sim=cosine_similarity):
    # initializing the empty list of recommended movies
    recommended_movies = []
    all_movies = []
    idx = []
    score_series = []
    top_10_movielist = []
    movies = movieList[1]
    # print(movieList[0])
    # print(len(movieList))
    # getting the index of the movie that matches the liked movies of user
    for i in range(0, len(movies)):
        idx.append(indices[indices == movies[i]].index[0])
    print("--idx--")
    print(idx)

    # creating Series for each movie in movielist with the similarity scores in descending order
    for j in range(0, len(movies)):
        # print("for loop")
        score_series.append(pd.Series(cosine_sim[idx[j]]).sort_values(ascending=False))
        # score_series[j] = score_series[j].iloc[1:11]
        print(score_series[j])
        # drop rows that contains identical movies with movielist provided
        for i in range(0, len(idx)):
            if idx[i] in score_series[j].index:
                # print("if")
                score_series[j] = score_series[j].drop(labels=idx[i])
        print("-----------------")
        print(score_series[j])
        # retrieving 10 suggested movie for each movie in movielist
        score_series[j] = score_series[j].iloc[0:10]
        print("-----------------")
        print(score_series[j])

    print("-----------")
    # combining suggested movies which are in seperate lists for each movie
    score_series = pd.concat(score_series).drop_duplicates().sort_values(ascending=False)
    print(score_series)
    # retreiving 10 movies with highest scores from combination
    top_10_movielist = list(score_series.iloc[0:10].index)
    print("-----------")
    print(len(top_10_movielist))
    print(top_10_movielist)

    # ---------------------------------------------------------------------

    # populating the list with the titles of the best 10 matching movies
    for j in top_10_movielist:
        recommended_movies.append(list(allData_df.index)[j])
    # print(recommended_movies)
    # recommended_movies = random.sample(recommended_movies,10)
    print("-----Recommended Movies----")
    return homepage_recommendations(recommended_movies)

    '''print(recommended_movies)
    print(len(recommended_movies))
    return recommended_movies'''









# api_key = config.tmdb_api_key # get TMDB API key from config.py file
# Flask implementaion
app = Flask(__name__)


# Configuring the firebase authentication and realtime database 
config = {
        # "apiKey":"AIzaSyCBNM_acw--K7T1DThsp2YUTlBtkntH4ek",
        # "authDomain": "movierecommend-d035f.firebaseapp.com",
        # "databaseURL": "https://movierecommend-d035f.firebaseio.com",
        # "projectId": "movierecommend-d035f",
        # "storageBucket": "movierecommend-d035f.appspot.com",
        # "messagingSenderId": "500565160192",
        # "appId": "1:500565160192:web:c878e153ca9fc2b8d583e5",
        # "measurementId": "G-DGJQFZCXNH"


        "apiKey": "AIzaSyAHZFNqy4_PXAMadTISoCAIvHsL63NKuu0",
        "authDomain": "movierecommendation-3aeec.firebaseapp.com",
        "databaseURL": "https://movierecommendation-3aeec.firebaseio.com",
        "projectId": "movierecommendation-3aeec",
        "storageBucket": "movierecommendation-3aeec.appspot.com",
        "messagingSenderId": "825351440204",
        "appId": "1:825351440204:web:86eca0a9d4d679e8c131e9",
        "measurementId": "G-4MS0NQ9K9P"
}

firebase = pyrebase.initialize_app(config)

auth = firebase.auth()
db = firebase.database()
userSignin = auth.sign_in_with_email_and_password('monica@test.com', 'test@123')
# userSignin = auth.sign_in_with_email_and_password(email, password)
user_refresh = auth.refresh(userSignin['refreshToken'])
currentUser=auth.get_account_info(userSignin['idToken'])
currentUserID=currentUser["users"][0]["localId"]


print("--Recommendations--")
movieList = []





# Retrieving popular movies in every genres by calling the homepage_movies function and storing in database

Horror=homepage_movies("Horror")
for i in range(len(Horror)):
    db.child(currentUserID).child("Horror").child(i).set(Horror[i])

Action=homepage_movies("Action")
for i in range(len(Horror)):
    db.child(currentUserID).child("Action").child(i).set(Action[i])
Adventure=homepage_movies("Adventure")
for i in range(len(Adventure)):
    db.child(currentUserID).child("Adventure").child(i).set(Adventure[i])
Drama=homepage_movies("Drama")
for i in range(len(Adventure)):
    db.child(currentUserID).child("Drama").child(i).set(Drama[i])
Comedy=homepage_movies("Comedy")

for i in range(len(Comedy)):
    db.child(currentUserID).child("Comedy").child(i).set(Comedy[i])
#Retrieving Recommended movies based on likes of the user
a = db.child(currentUserID).child('Likes').get()
if a.val() is not None:
    print(a)

    print('--------sasdavefw')
    print(a.val())
    tuple_list = a.val()
    print(list(tuple_list.items()))
    print(list(tuple_list))
    # using list comprehension to
    # perform Unzipping
    test_list = list(tuple_list.items())
    movieList = [[ i for i, j in test_list ],
        [ j for i, j in test_list ]]
    print(movieList[1][0])
    Recommendation=recommendations(movieList)
    for i in range(len(Recommendation)):
        db.child(currentUserID).child("Recommendation").child(i).set(Recommendation[i])
# else:
#     Recommendation=[]

def stream_handler(message):
     # put
    print(message["event"])
    
    print(message["path"])
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
    a = db.child(currentUserID).child('Likes').get()
    # print(a)

    print('--------sasdavefw')
    print(a.val())
    tuple_list = a.val()
    print(list(tuple_list.items()))
    print(list(tuple_list))
    # using list comprehension to
    # perform Unzipping
    test_list = list(tuple_list.items())
    movieList = [[ i for i, j in test_list ],
        [ j for i, j in test_list ]]
    print(movieList[1][0])
    # Calling the recommendation function
    Recommendation=recommendations(movieList)
     
    for i in range(len(Recommendation)):
        db.child(currentUserID).child("Recommendation").child(i).set(Recommendation[i])
    
    


my_stream = db.child(currentUserID).child("Likes").stream(stream_handler)
# Retrievong data from the database and storing it as list of dictionaries called row_name and passing it to the html page
b=db.child(currentUserID).child('Recommendation').get()
if b.val() is None:

    row_name=[{'Adventure':Adventure,'Horror':Horror,'Drama':Drama, 'Action':Action,'Comedy':Comedy}]
else:
    row_name=[{'Recommendation':Recommendation,'Adventure':Adventure,'Horror':Horror,'Drama':Drama, 'Action':Action,'Comedy':Comedy}]
    print(Recommendation)
# Calling html main page
@app.route('/home')
def main_page():
    return render_template('main.html',row_name=row_name,db=db,currentUserID=currentUserID)
# Calling html like page
@app.route('/like/<int:movieid>/<moviename>')
def like(moviename,movieid):
    return render_template('like.html',movieid=movieid,moviename=moviename,row_name=row_name,db=db,currentUserID=currentUserID)
# Calling html liked page 
@app.route('/liked_page/<int:id>/<name>')
def liked_page(name,id):
    return render_template('liked_page.html',id=id,name=name,row_name=row_name,db=db,currentUserID=currentUserID)
# main
if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True 
     
    app.run(debug=True) 