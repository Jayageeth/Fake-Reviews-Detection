import os
import pickle, nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

def countvectorize(statement):
    countvectorizer = pickle.load(open("countvectorizer.sav", 'rb'))
    statement = countvectorizer.transform(statement).toarray()
    return statement


def onehotencode(rating, verified_purchase, product_category, X):
    labelencoder_1 = pickle.load(open("labelencoder_1.sav", 'rb'))
    labelencoder_2 = pickle.load(open("labelencoder_2.sav", 'rb'))
    labelencoder_3 = pickle.load(open("labelencoder_3.sav", 'rb'))

    ct1 = pickle.load(open("columntransformer1.sav", 'rb'))
    ct2 = pickle.load(open("columntransformer2.sav", 'rb'))
    ct3 = pickle.load(open("columntransformer3.sav", 'rb'))

    w, h = 3, 1;
    new_col = [[0 for x in range(w)] for y in range(h)]
    num = 0

    for i in range(0, 1):
        new_col[i][0] = rating
        new_col[i][1] = verified_purchase
        new_col[i][2] = product_category
	
    new_col = np.array(new_col)

    new_col[:, 0] = labelencoder_1.transform(new_col[:, 0])
    new_col[:, 1] = labelencoder_2.transform(new_col[:, 1])
    new_col[:, 2] = labelencoder_3.transform(new_col[:, 2])
	
    new_col = ct1.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)
	
    new_col = ct2.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)
	
    new_col = ct3.transform(new_col)
    try:
        new_col = new_col.toarray()
    except:
        #Do Nothing
        pass
    new_col = new_col.astype(np.float64)
	
    X= np.append(X, new_col, axis=1)
    return X

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


def postag(sentence, X):
    w, h = 2, 1;
    pos_tag = [[0 for x in range(w)] for y in range(h)]
    num = 0

    sentence = POS_Tagging(sentence)

    if sentence=='T':
        pos_tag[0][0] = 1
        pos_tag[0][1] = 0
    else:
        pos_tag[0][0] = 0
        pos_tag[0][1] = 1

    X = np.append(X, pos_tag, axis=1)
    return X


def classify(X):
    bernoullinb = pickle.load(open("bernoullinb.sav", 'rb'))
    return bernoullinb.predict(X)

def get_result(statement, rating, verified_purchase, product_category):
    X = countvectorize([statement])
    X = postag(statement, X)
    X = onehotencode(rating, verified_purchase, product_category, X)
	
    #print (X[0])
	
    X = classify(X)
    return X

if __name__ == '__main__':
	print("\n---------------------------------------------------------------------------------------\n")
	review_text = input("Enter your Review : ")
	product_rating = input("Enter your Product Rating (On a scale of 1 to 5) : ")
	verified_purchase = input("Enter if it's a Verified Purchase (Y or N) : ")
	product_category = input("Enter your Product Category (Apparel, Clothing or Shoes) : ")

	answer = get_result(review_text, product_rating, verified_purchase, product_category)

	if answer == 1:
		print ("It is a True Review")

	else:
		print ("It is a False Review")
