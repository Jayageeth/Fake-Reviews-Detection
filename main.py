import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import codecs
from progressBar import printProgressBar

#############################################################################################################
# Read Dataset
#############################################################################################################

# Dataset Download Link : https://www.kaggle.com/lievgarcia/amazon-reviews
with codecs.open("amazon_dataset_1.csv", "r",encoding='utf-8', errors='ignore') as file_dat:
     dataset = pd.read_csv(file_dat)

len_dataset = math.floor(len(dataset)/1)

y = dataset.iloc[:,1:2].values

#############################################################################################################
# Download nltk Libraries
#############################################################################################################

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

print("\n---------------------------------------------------------------------------------------\n")

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
num = 0

#############################################################################################################
# Tokenization and Stemming
#############################################################################################################

print ("\nPerforming Tokenization and Stemming.")
for i in range(0, math.floor(len_dataset)):
    review=re.sub('[^a-zA-Z]',' ',dataset['REVIEW_TEXT'][i])
    review=review.lower()
    review=review.split()
    #print (review)
    review=[word for word in review if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

    printProgressBar(iteration = num, total = len_dataset, prefix = 'Progress:', suffix = 'Complete', length = 50)
    num = num + 1

#############################################################################################################
# Count Vectorization
#############################################################################################################

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:len_dataset,1]

filename = 'countvectorizer.sav'
pickle.dump(cv, open(filename, 'wb'))

#############################################################################################################
# POS Tagging
#############################################################################################################

def POS_Tagging(sentence):
    tagged_list = []
    tags = []
    count_verbs = 0
    count_nouns = 0
    text=nltk.word_tokenize(sentence)
    tagged_list = (nltk.pos_tag(text))

    tags = [x[1] for x in tagged_list]
    for each_item in tags:
        if each_item in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            count_verbs+=1
        elif each_item in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            count_nouns+=1
        else:
            continue
    if count_verbs > count_nouns:
        sentence = 'F'
    else:
        sentence = 'T'

    return sentence

w, h = 2, len_dataset;
pos_tag = [[0 for x in range(w)] for y in range(h)]
num = 0

print ("\n\nPerforming POS Tagging.")
for i in range(0,len_dataset):

    text = dataset['REVIEW_TEXT'][i]
    sentence=POS_Tagging(text)

    printProgressBar(iteration = num, total = len_dataset, prefix = 'Progress:', suffix = 'Complete', length = 50)
    num = num + 1

    if sentence=='T':
        pos_tag[i][0] = 1
        pos_tag[i][1] = 0
        #X[i].insert(1)
        #X[i].insert(0)
    else:
        pos_tag[i][0] = 0
        pos_tag[i][1] = 1

    #print (pos_tag[i])
        #X[i].insert(0)
        #X[i].insert(1)
X= np.append(X, pos_tag, axis=1)

#############################################################################################################
# Label Encoding
#############################################################################################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
y = le.fit_transform(y)

w, h = 3, len_dataset;
new_col = [[0 for x in range(w)] for y in range(h)]
num = 0

test = dict()
test_num = 0

for i in range(0, len_dataset):
    new_col[i][0] = dataset["RATING"][i]
    new_col[i][1] = dataset["VERIFIED_PURCHASE"][i]
    new_col[i][2] = dataset["PRODUCT_CATEGORY"][i]

    if new_col[i][2] not in test.keys() :
        test[new_col[i][2]] = 1
        test_num = test_num + 1

        #print (new_col[i][2])

#print (test_num)

new_col = np.array(new_col)

labelEncoder = LabelEncoder()
new_col[:, 0] = labelEncoder.fit_transform(new_col[:, 0])
filename = 'labelencoder_1.sav'
pickle.dump(labelEncoder, open(filename, 'wb'))

new_col[:, 1] = labelEncoder.fit_transform(new_col[:, 1])
filename = 'labelencoder_2.sav'
pickle.dump(labelEncoder, open(filename, 'wb'))

new_col[:, 2] = labelEncoder.fit_transform(new_col[:, 2])
filename = 'labelencoder_3.sav'
pickle.dump(labelEncoder, open(filename, 'wb'))

#############################################################################################################
# OneHotEncoder / Column Transformer
#############################################################################################################

ct1 = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
new_col = ct1.fit_transform(new_col)
new_col = new_col.astype(np.float32)
filename = 'columntransformer1.sav'
pickle.dump(ct1, open(filename, 'wb'))

'''
onehotencoder = OneHotEncoder(categorical_features = [0])
new_col = onehotencoder.fit_transform(new_col).toarray()
filename = 'onehotencoder1.sav'
pickle.dump(onehotencoder, open(filename, 'wb'))
'''

ct2 = ColumnTransformer([("Country", OneHotEncoder(), [5])], remainder = 'passthrough')
new_col = ct2.fit_transform(new_col)
new_col = new_col.astype(np.float32)
filename = 'columntransformer2.sav'
pickle.dump(ct2, open(filename, 'wb'))

'''
onehotencoder = OneHotEncoder(categorical_features = [5])
new_col = onehotencoder.fit_transform(new_col).toarray()
filename = 'onehotencoder2.sav'
pickle.dump(onehotencoder, open(filename, 'wb'))
'''

ct3 = ColumnTransformer([("Country", OneHotEncoder(), [7])], remainder = 'passthrough')
new_col = ct3.fit_transform(new_col)
new_col = new_col.toarray()
new_col = new_col.astype(np.float32)
filename = 'columntransformer3.sav'
pickle.dump(ct3, open(filename, 'wb'))

'''
print (X)
print ("***************************************************************************")
print (new_col.astype(int))
'''

'''
onehotencoder = OneHotEncoder(categorical_features = [7])
new_col = onehotencoder.fit_transform(new_col).toarray()
filename = 'onehotencoder3.sav'
pickle.dump(onehotencoder, open(filename, 'wb'))
'''

new_col = new_col.astype(np.uint8)
X = X.astype(np.uint8)
X = np.append(X, new_col, axis=1).astype(np.uint8)

#############################################################################################################
# Split in Train and Test Set
#############################################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

#############################################################################################################
# Training Classifiers
#############################################################################################################

print ("\n\nTraining Classifier on Bernoulli Naive Bayes.")
from sklearn.naive_bayes import BernoulliNB

classifier=BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

filename = 'bernoullinb.sav'
pickle.dump(classifier, open(filename, 'wb'))

from sklearn.metrics import accuracy_score
print ("\nAccuracy of Bernoulli Naive Bayes is : ")
print (accuracy_score(y_test, y_pred) * 100)

print ("\n\nTraining Classifier on Support Vector Machine.")
from sklearn.svm import SVC # "Support Vector Classifier"
clf = SVC(kernel='rbf')
# fitting x samples and y classes
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print ("\nAccuracy of Support Vector Machine is : ")
print(accuracy_score(y_test, y_pred) * 100)
