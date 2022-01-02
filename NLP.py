# import libraries
print("importing libraries...")
import pandas as pd   
import numpy as np
import nltk
pd.options.mode.chained_assignment = None #suppress warnings

nltk.download('wordnet')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# spacy.load('en_core_web_md')
import en_core_web_md

print("setting up dataframe...")
# setting up dataframe
data_file  = 'yelp_final.csv'
yelp_full = pd.read_csv(data_file)
needed_columns = ['text', 'stars'] 
yelp = yelp_full[needed_columns]

def is_good_review(num_stars):
    return num_stars > 3

yelp['is_good_review'] = yelp['stars'].apply(is_good_review)

print("setting up model...")
# setting up x and y for classification
X_text = yelp['text']
y = yelp['is_good_review']

text_to_nlp = en_core_web_md.load() #Prepare Spacy

def tokenize_vecs(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): 
          # -PRON- is a special all inclusive "lemma" spaCy uses for any pronoun, we want to exclude these 
            clean_tokens.append(token)
    return clean_tokens

def bagWords(text):
    review = tokenize_vecs(text) # returns cleaned list of spacy tokens
    review_vec = [0]*300
    for word in review:
        review_vec += word.vector
    review_vec = review_vec / len(review)
    return review_vec

X_word2vec = []
for text in X_text:
  review_vec = bagWords(text)
  X_word2vec.append(review_vec)
  
X_word2vec = np.array(X_word2vec)

print("training model...")
# classification using log regression
w2v_model = LogisticRegression()
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(X_word2vec, y, test_size=0.2, random_state=101)
w2v_model.fit(X_train_word2vec, y_train_word2vec)

w2v_preds = w2v_model.predict(X_test_word2vec) 
accuracy = accuracy_score(y_test_word2vec, w2v_preds)

print("training done! predicted accuracy:", accuracy)

while True:
    curRev = input("Input a review for classification! (-1 to exit)")
    if curRev == "-1":
        print("bye!")
        break
    else:
        prediction = w2v_model.predict([bagWords(curRev)])
        if prediction:
            print ("This was a GOOD review!")
        else:
            print ("This was a BAD review!")