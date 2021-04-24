import keras
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df_responses = pd.read_csv('../data/response_format_cleaned_ds1.csv', sep=';', header=0)
df_responses.drop(df_responses.columns[[0]], axis=1, inplace=True)

"""
Using 5-gram contexts from the database, a co-occurrence (word by word) matrix was set up, 
where the rows contained the 120,000 most common words in the n-gram database and the columns 
consisted of the 10,000 most common words in the n-gram database.

The variable 'space' is a matrix of the semantic space with dimentions reduced to 512.
"""
df_space = pd.read_csv('../data/spaceEnglish1.csv', encoding= 'unicode_escape')
df_space.set_index('words', inplace=True)
df_space.drop(df_space.columns[[0]], axis=1, inplace=True)
df_space.dropna(inplace=True)

SPACE_MEAN = pd.Series.to_numpy(df_space.mean())
WORDS_IN_SPACE = set(df_space.index.values)
SPACE_COLS = list(df_space.columns)


def aggregating_words(responses):
    """
    Controlling for artifacts relating to frequently occurring words.

    1) Calculate, from Google N-gram, a frequency weighted average of all semantic representations in the space.
    (So that the weighting is proportional to how frequently the words occur in Google N-gram.)
    2) Subtract this mean prior to aggregating each word, and then add to the final value.
    """
    res_arr = np.zeros(512)
    for word in responses:
        word_arr = pd.Series.to_numpy(df_space.loc[word])
        res_arr = res_arr + (word_arr - SPACE_MEAN)
    
    res_arr += SPACE_MEAN    
    res_arr = res_arr / res_arr.sum() # Normalizing aggregated vector
    return res_arr


def clean_text(text):
    """
    Cleans the string from punctuations and removes all words which are not represented in the semantic space. 
    """
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = list(set(text.split()))
        # TODO: Hantera ord som inte finns i spacet. Nu ignoreras dem.
        cleaned_words = [w for w in text if w in WORDS_IN_SPACE] 
        return cleaned_words
    
    except Exception as e: 
        print(e)

def extract_words(inx):
    """
    For one row extracts the string of words/text from the important columns. 
    The strings are cleaned using the clean_text function and a list of all words (without duplicates) is returned.
    """
    dep_important_columns = ['Deptext','dep_all_phraces', 'dep_all_words',  'dep_all_selected1']
    wor_important_columns = ['Wortext', 'wor_all_phraces', 'wor_all_words',  'wor_all_selected1']
    
    all_responded_words_dep = []
    all_responded_words_wor = []
    res = df_responses.iloc[inx]
    
    for column_name in dep_important_columns:
        words_in_column = res[column_name]
        if isinstance(words_in_column, str): 
            words = clean_text(words_in_column)
            for word in words:
                if word not in all_responded_words_dep:
                    all_responded_words_dep.append(word)
    
    for column_name in wor_important_columns:
        words_in_column = res[column_name]
        if isinstance(words_in_column, str): 
            words = clean_text(words_in_column)
            for word in words:
                if word not in all_responded_words_wor:
                    all_responded_words_wor.append(word)
    
    return np.array(all_responded_words_dep), np.array(all_responded_words_wor)


"""
Goes through all rows in response dataframe and for each person, converts all the words to an aggregated vector
and adds this to a dictionary to later be converted to a dataframe. 
"""
all_responses_space_dep = {}
all_responses_space_anx = {}

for i in range(len(df_responses.index)):
    words_dep, words_anx = extract_words(i)
    try:
        all_responses_space_dep[i] = aggregating_words(words_dep)
        all_responses_space_anx[i] = aggregating_words(words_anx)
    except Exception as e:
        print(e)


"""
Creates a dataframe with dimension (nbr of participants x 512).
Each row stores the aggregated semantic vector representations of one persons responses.

This matrix can then be used for the analysis steps. 
"""
vectors_dep = list(all_responses_space_dep.values())
vectors_anx = list(all_responses_space_anx.values())
response_space_dep = pd.DataFrame(vectors_dep, columns = SPACE_COLS) 
response_space_anx = pd.DataFrame(vectors_anx, columns = SPACE_COLS) 