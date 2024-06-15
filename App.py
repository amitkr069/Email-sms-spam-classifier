import streamlit as st
import pickle
#import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# ek function bnayenge and yahi sara kaam kar dega

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text) #tokenize krne ke liye- words me convert kr dega

  #for removing punctuation and special characters
  new = []
  for i in text:
    if i.isalnum():
      new.append(i)

  #remvoving stop words
  text = new[:]
  new.clear()
  for i in text:
    if i not in stopwords.words('english'):
      new.append(i)

  #stemming
  text = new[:]
  new.clear()
  #ps = nltk.PorterStemmer()
  for i in text:
    new.append(ps.stem(i))

  return " ".join(new) #as a string return karenge

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    #steps:
    # 1. Preprocess
    # 2. Vectorize
    # 3. predict
    # 4. Display

    #preprocess wla function is same as the goole colab me jo bnaye the
    transformed_sms  = transform_text(input_sms)

    #vectorize
    vectorize = tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vectorize)[0]

    #display
    if result == 0:
        st.header("Not Spam")
    else:
        st.header("Spam")