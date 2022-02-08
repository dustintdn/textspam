# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:46:07 2022

@author: dusti
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt


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

##-----

messages['label'].value_counts()
#data is very imbalanced

#resampling hams to match spams shape
hams = hams.sample(spams.shape[0])
data = hams.append(spams, ignore_index=True)

###HISTOGRAMS###

#character length (message length)
plt.hist(data[data['label']=='ham']['length'], bins = 100, alpha=0.7)
plt.hist(data[data['label']=='spam']['length'], bins = 100, alpha=0.7)
plt.xlabel('SMS Character Length')
plt.ylabel('Frequency')
plt.legend(['hams', 'spams'],loc="upper right")
plt.show()

#punctuation
plt.hist(data[data['label']=='ham']['punctuation'], bins = 100, alpha=0.7)
plt.hist(data[data['label']=='spam']['punctuation'], bins = 100, alpha=0.7)
plt.xlabel('SMS Punctuation')
plt.ylabel('Frequency')
plt.legend(['hams', 'spams'],loc="upper right")
plt.show()

###WORD CLOUDS###
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

spams = data[data['label'] == 'spam']
hams = data[data['label'] == 'ham']

#hams word cloud
word_cloud_ham = ''.join(hams['text'])
word_cloud1 = WordCloud(max_font_size=100,
                       max_words=100,
                       background_color="white",
                       scale = 10,
                       width=800,
                       height=400
                       ).generate(word_cloud_ham)
plt.figure()
plt.imshow(word_cloud1,
           interpolation="bilinear")
plt.axis("off")
plt.show()

#spam word cloud
word_cloud_spam = ''.join(spams['text'])
word_cloud2 = WordCloud(max_font_size=100,
                       max_words=100,
                       background_color="white",
                       scale = 10,
                       width=800,
                       height=400
                       ).generate(word_cloud_spam)
plt.figure()
plt.imshow(word_cloud2,
           interpolation="bilinear")
plt.axis("off")
plt.show()

