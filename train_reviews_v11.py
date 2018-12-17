import pandas as pand
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import matutils
from gensim import corpora
from gensim import models
from sklearn import model_selection
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

trn_neg_lst = os.listdir("train_negative")
trn_pos_lst = os.listdir("train_positive")

def crt_train_df(trn_file_lst,trn_dir,review_type):
    rev_lab = []
    rev_desc = []
    rev_act =[]
    for j in trn_file_lst:
        rev_lab.append(review_type)
        desc = str(open(trn_dir + '/' + j).read())
        rev_desc.append(desc)
        rev_act.append(str(j.split('_')[0]))
    data = pand.DataFrame({'Review Label':rev_lab,'Review Desc':rev_desc,'Review Actual':rev_act})
    return data

neg_df = crt_train_df(trn_neg_lst,'train_negative','negative')
pos_df = crt_train_df(trn_pos_lst,'train_positive','positive')

review_score = []
for i in pos_df.index:
    if ((pos_df['Review Label'][i] == 'positive') & (pos_df['Review Actual'][i] == 't')):
        review_score.append(10)
    elif ((pos_df['Review Label'][i] == 'positive') & (pos_df['Review Actual'][i] == 'd')):
        review_score.append(-10)
    else:
        print('Error!')
pos_df['score'] = review_score

review_score = []
for i in neg_df.index:
    if ((neg_df['Review Label'][i] == 'negative') & (neg_df['Review Actual'][i] == 't')):
        review_score.append(20)
    elif ((neg_df['Review Label'][i] == 'negative') & (neg_df['Review Actual'][i] == 'd')):
        review_score.append(-20)
    else:
        print('Error!')
neg_df['score'] = review_score

review_data = pos_df.merge(neg_df,how='outer')
review_data = review_data[['Review Desc','score']]

def extract_bow(df):
    review_tokenized = []
    lmt = WordNetLemmatizer()
    for index, datapoint in df.iterrows():
        tokenize_words = word_tokenize(datapoint["Review Desc"].lower(),language='english')
        pos_word = pos_tag(tokenize_words)
        tokenize_words = ["_".join([lmt.lemmatize(i[0]),i[1]]) for i in pos_word if (i[0] not in stopwords.words("english") and len(i[0]) > 2)]
        review_tokenized.append(tokenize_words)
    df["review_tokenized"] = review_tokenized
    return df

data = extract_bow(review_data)

def vectorize_comments(df):
    d = corpora.Dictionary(df["review_tokenized"])
    d.filter_extremes(no_below=2, no_above=0.8)
    d.compactify()
    corpus = [d.doc2bow(text) for text in df["review_tokenized"]]
    corpus = matutils.corpus2csc(corpus, num_terms=len(d.token2id))
    corpus = corpus.transpose()
    return d, corpus

dictionary,corpus = vectorize_comments(data)

def train_rfc(X,y):
    n_estimators = [100]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]
    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,'min_samples_split': min_samples_split}
    clf = GridSearchCV(RandomForestClassifier(verbose=1,n_jobs=-1), cv=4, param_grid=parameters)
    clf.fit(X, y)
    return clf



X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus, data["score"], test_size=0.6, random_state=2016)
rfc_clf = train_rfc(X_train,y_train)
review = []
review = rfc_clf.predict(corpus)
rating_pos=0
rating_neg=0
tot=0
frat=0
for i in review :
    if (i==10) :
        rating_pos = rating_pos + 1
        tot=tot+1
    if (i==20) :
        rating_neg = rating_neg + 1
        tot=tot+1

frat=(rating_pos + rating_neg)/tot
print ("Original rating : 2.3")
print ("New Rating after eliminating fake review : " , frat)
print ("Accuracy of RF  :{}".format(rfc_clf.best_score_))