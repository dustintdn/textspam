# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:29:48 2022

@author: dusti
"""
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

messages = pd.read_csv('data.csv')

#instantiate lemmatizer
lemmatizer = WordNetLemmatizer()
corpus = []

#count lengths and punctuations
length=[]
for i in range(len(messages)):
    length.append(len(messages["text"][i]))
    
messages["length"]=length


count=0
punct=[]

for i in range(len(messages)):
    for j in messages['text'][i]:
        if j in string.punctuation:
            count+=1
    punct.append(count)
    count=0
    
messages["punctuation"]=punct

#lemmatization and de-stopwords-ing
for i in range(0,len(messages)):
    words = re.sub('[^a-zA-Z]',' ',messages['text'][i])
    words = words.lower()
    words = words.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words = ' '.join(words) 
    corpus.append(words)
 
#spam-ham split    
messages["text"]=corpus
spams = messages[messages['label'] == 'spam']
hams = messages[messages['label'] == 'ham']

messages['label'].value_counts()
#data is very imbalanced

#resampling hams to match spams shape
hams = hams.sample(spams.shape[0])
data = hams.append(spams, ignore_index=True) 


#-----------------------------
###VADER####

hams_sentiment = pd.DataFrame()
hams_sentiment['neg'] = 0
hams_sentiment['neu'] = 0
hams_sentiment['pos'] = 0
hams_sentiment['compound'] = 0


for word in hams['text']:
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(word)
    hams_sentiment = hams_sentiment.append(sentiment_dict, ignore_index=True, sort=False)
    
spams_sentiment = pd.DataFrame()
spams_sentiment['neg'] = 0
spams_sentiment['neu'] = 0
spams_sentiment['pos'] = 0
spams_sentiment['compound'] = 0

for word in spams['text']:
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(word)
    spams_sentiment = spams_sentiment.append(sentiment_dict, ignore_index=True, sort=False)

#Creating labels for hams_sentiment
conditions = [
    (hams_sentiment['compound'] >= 0.05),
    (hams_sentiment['compound'] <= - 0.05),
    (hams_sentiment['compound'] > -0.05) & (hams_sentiment['compound'] < 0.05) 
    ]

# create a list of the values we want to assign for each condition
values = ['Positive', 'Negative', 'Neutral']

# create a new column and use np.select to assign values to it using our lists as arguments
hams_sentiment['label'] = np.select(conditions, values)


#Creating labels for spams_sentiment
conditions = [
    (spams_sentiment['compound'] >= 0.05),
    (spams_sentiment['compound'] <= - 0.05),
    (spams_sentiment['compound'] > -0.05) & (spams_sentiment['compound'] < 0.05) 
    ]

# create a list of the values we want to assign for each condition
values = ['Positive', 'Negative', 'Neutral']

# create a new column and use np.select to assign values to it using our lists as arguments
spams_sentiment['label'] = np.select(conditions, values)
