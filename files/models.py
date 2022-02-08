# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:36:40 2022

@author: dusti
"""
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#import using data.csv
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


#splitting my balanced data
X_train , X_test , y_train , y_test = train_test_split(data['text'], data['label'], 
                                                       random_state=77, shuffle=True, stratify=data['label'])
#tfidf vectorizer
tfidf=TfidfVectorizer()
X_train_tfidf_vect=tfidf.fit_transform(X_train).toarray()
X_train_tfidf_vect.shape

#count single word tokenize
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_train_count_vect=count_vect.fit_transform(X_train).toarray()
X_train_count_vect.shape

#countvect BIGRAMS
from sklearn.feature_extraction.text import CountVectorizer
cv2=CountVectorizer(ngram_range=(1,2))
X_train_cv2=cv2.fit_transform(X_train).toarray()
X_train_cv2.shape


#Logistic Regression
from sklearn.linear_model import LogisticRegression
text_log=Pipeline([('cv2',cv2),('log',LogisticRegression())])
text_log.fit(X_train,y_train)
y_preds_log=text_log.predict(X_test)

text_log.score(X_train,y_train)
text_log.score(X_test,y_test)

print(confusion_matrix(y_test,y_preds_log))
print(classification_report(y_test,y_preds_log))

#SVM (SVclassifier)
from sklearn.svm import LinearSVC
text_svm=Pipeline([('tfidf',TfidfVectorizer()),('svm',LinearSVC(C=1))])
text_svm.fit(X_train,y_train)
y_preds_svm=text_svm.predict(X_test)

text_svm.score(X_train,y_train)
text_svm.score(X_test,y_test)

print(confusion_matrix(y_test,y_preds_svm))
print(classification_report(y_test,y_preds_svm))

#gridsearch LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
model = LinearSVC()

param_grid = {'C': [1, 5, 10, 50],}
grid = GridSearchCV(model, param_grid)

text_grid = Pipeline([('tfidf',TfidfVectorizer()),('grid', grid)])
text_grid.fit(X_train, y_train)
text_grid.score(X_test,y_test)

print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

#gridsearch SVC
from sklearn.svm import SVC
model = SVC()
param_grid = {'C':[1,10,100,1000],
              'gamma':[1,0.1,0.001,0.0001]}
grid = GridSearchCV(model, param_grid)
text_grid = Pipeline([('tfidf',TfidfVectorizer()),('grid', grid)])
text_grid.fit(X_train, y_train)
text_grid.score(X_test,y_test)

print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

y_preds_grid=text_grid.predict(X_test)

text_grid.score(X_train,y_train)
text_grid.score(X_test,y_test)

text_svm=Pipeline([('tfidf',TfidfVectorizer()),('svm',SVC(C=10, gamma=0.1))])
text_svm.fit(X_train,y_train)
y_preds_svm=text_svm.predict(X_test)

text_svm.score(X_train,y_train)
text_svm.score(X_test,y_test)

print(confusion_matrix(y_test,y_preds_svm))
print(classification_report(y_test,y_preds_svm))


#XGB
import xgboost as xgb 
#dtrain = xgb.dmatrix(X_train, label=y_train)
#export C:/Users/dusti/anaconda3/Lib/site-packages/xgboost/python-package
#dtest = xgb.dmatrix(X_test, label=y_test)

xg_cl = xgb.XGBClassifier(objective='binary:logistic',seed=111)

#pipelined xgb
text_xgb=Pipeline([('tfidf',TfidfVectorizer()),('xgb',xg_cl)])
text_xgb.fit(X_train,y_train)
y_preds_xgb=text_xgb.predict(X_test)

text_xgb.score(X_train,y_train)
text_xgb.score(X_test,y_test)

print(confusion_matrix(y_test,y_preds_xgb))
print(classification_report(y_test,y_preds_xgb))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
param_grid = { 
    'n_estimators': [50,100,150],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

grid = GridSearchCV(model, param_grid)
text_grid = Pipeline([('tfidf',TfidfVectorizer()),('random', grid)])
text_grid.fit(X_train, y_train)
text_grid.score(X_test,y_test)

print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)

text_rfc=Pipeline([('cv2',cv2),('rfc',RandomForestClassifier(criterion='gini', max_depth=8, max_features='auto', n_estimators=150))])
text_rfc.fit(X_train,y_train)
y_preds_log=text_log.predict(X_test)

text_rfc.score(X_train,y_train)
text_rfc.score(X_test,y_test)

print(confusion_matrix(y_test,y_preds_log))
print(classification_report(y_test,y_preds_log))


from sklearn.model_selection import cross_val_score

log_acc = cross_val_score(text_log, X_test, y_test, scoring='accuracy', cv=5).mean()
svm_acc = cross_val_score(text_svm, X_test, y_test, scoring='accuracy', cv=5).mean()
xgb_cc = cross_val_score(text_xgb, X_test, y_test, scoring='accuracy', cv=5).mean()
rfc_acc = cross_val_score(text_rfc, X_test, y_test, scoring='accuracy', cv=5).mean()

print("Random Forest Classifier Accuracy: " + str(rfc_acc))
print("L2 Penalized Logistic Regression Accuracy: " + str(log_acc))
print("Support Vector Classifier Accuracy: " + str(svm_acc))
print("XGBoosted Decision Tree Accuracy: " + str(xgb_cc))


#plotting coefficients from SVM
#magnitude of words between spam and legit texts
import matplotlib.pyplot as plt
import numpy as np
vect = TfidfVectorizer().fit(X_train)

def visualize_coefficients(coefficients, feature_names, n_top_features=25):
    """Visualize coefficients of a linear model.
    Parameters
    ----------
    coefficients : nd-array, shape (n_features,)
        Model coefficients.
    feature_names : list or nd-array of strings, shape (n_features,)
        Feature names for labeling the coefficients.
    n_top_features : int, default=25
        How many features to show. The function will show the largest (most
        positive) and smallest (most negative)  n_top_features coefficients,
        for a total of 2 * n_top_features coefficients.
    """
    coefficients = coefficients.squeeze()
    if coefficients.ndim > 1:
        # this is not a row or column vector
        raise ValueError("coeffients must be 1d array or column vector, got"
                         " shape {}".format(coefficients.shape))
    coefficients = coefficients.ravel()

    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients {} doesn't match number of"
                         "feature names {}.".format(len(coefficients),
                                                    len(feature_names)))
    # get coefficients with large absolute values
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients,
                                          positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ['#ff2020' if c < 0 else '#0000aa'
              for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients],
            color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features),
               feature_names[interesting_coefficients], rotation=60,
               ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")
    

coefs = SVC(C=10, gamma=0.1, kernel='linear').fit(X_train_tfidf_vect,y_train).coef_
feature_names = vect.get_feature_names()


visualize_coefficients(coefs, feature_names, n_top_features=15)
