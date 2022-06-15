import pandas as pd
import numpy as np
import re
import itertools
import tqdm

import seaborn as sns
import tqdm
import matplotlib.pyplot as plt
%matplotlib inline

from bs4 import BeautifulSoup
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer

import nltk
nltk.download('stopwords')
nltk.download('words')
words = set(nltk.corpus.words.words())
words = set([w.lower() for w in words])

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download("wordnet")

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


from nltk.tokenize import sent_tokenize

import gensim
from gensim.downloader import load
from gensim.models import Word2Vec
w2v_model = gensim.downloader.load('word2vec-google-news-300')

from typing import Dict, List, Optional, Tuple



def data_processing(df_movie: pd.DataFrame, 
                          df_users: pd.DataFrame, 
                          df_rating: pd.DataFrame
) -> List[pd.DataFrame]:
    
    df_movies['clean_title'] = df_movies.title.apply(lambda x : procces_text(clean_text(x)))
    df_movies.drop("title", axis = 1, inplace = True)
    df_movies_clean = pd.concat([df_movies.drop("clean_title", axis=1), 
          pd.DataFrame(df_movies.clean_title.apply(string_embedding).to_list(), columns = ['w2v_' + str(i) for i in range(300)])], axis = 1)
    
    df_history = pd.merge(df_rating[['user_Id', 'movie_Id']], df_movies_clean.movie_Id, on = 'movie_Id', how = 'left')
    df_users_vec = group_w2v(df_history, df_movies_full)
    df_users_full = pd.merge(df_users, df_users_vec, on='user_Id')
    df_users_clean = df_users_full.rename(columns = {i: 'w2v_' + str(i) for i in range(300)})
    
    return [df_movies_clean, df_users_clean, df_rating_clean]
 

def clean_text(text: str) -> str:
    """
    Cleaning and preprocessing of the tags text with help of regular expressions.
    
    Arguments:
    --text: initial text.
    
    Return:
    --text: cleaned text.
        
    """
    text = re.sub("[^a-zA-Z]", " ",text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.lower()

    return text


def procces_text(text):
    """
    Processing tag text: tokenization, removing stop words, lemmatization.
    
    Arguments:
    --text: cleared and preprocessed text.
    
    Return:
    --text: processed text.
    """
    lemmatizer = WordNetLemmatizer() 

    text = [word for word in nltk.word_tokenize(text) if not word in stop_words]
    text = [lemmatizer.lemmatize(token) for token in text]
    text = [word for word in text if word in words]

    text = " ".join(text)
    
    return text

def string_embedding(string: str) -> np.ndarray:
    """
    Processing each word in the string with word2vec and return their aggregation (mean).
    
    Arguments:
    --string: string of tags.
    
    Return:
    --vec: vector of string embedding.
    """
    
    arr = string.split(' ')
    vec = 0
    cnt = 0
    for i in arr:
        try:
            vec += w2v_model[i]
            cnt += 1
        except:
            pass
    if cnt == 0:
        vec = np.zeros((300, 1))
    else:
        vec /= cnt
    return vec

def group_w2v(history: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregation (mean) embedded data for users watch history partitions.
    
    Arguments:
    --history: data frame of users movies history.
    --movies: data frame of movies. 
    
    Return:
    --df: data frame of users with aggregation embedded movies data.
    """
    users_id_arr = history.user_Id.unique()
    
    id_arr = []
    vec_arr = np.zeros((len(users_id_arr), 300))
    
    for user_id in tqdm.tqdm_notebook(range(len(users_id_arr))):
        vec = np.asarray(movies[movies.movie_Id.isin(history[history.user_Id == users_id_arr[user_id]].movie_Id)].iloc[:, 3:]).mean(axis=0) 
        
        id_arr.append(users_id_arr[user_id])
        vec_arr[user_id] = vec
    
    df = pd.DataFrame(vec_arr)
    df['user_Id'] = id_arr
    
    return df
