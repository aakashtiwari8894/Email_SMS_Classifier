import streamlit as st
import pickle
from PIL import Image
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def clean_text(text):
    text=text.lower()
    text =nltk.word_tokenize(text)
    y=[]
    for c in text :
        if c.isalnum():
            y.append(c)
    text=y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation and c not in stopwords.words("english"):
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf=pickle.load(open("vectorizer.pkl", "rb"))
model=pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")
img= Image.open("spam_img.png")
st.image(img, width=700)

input_sms= st.text_area("Enter The Message")
if st.button("Classify"):

#preprocessing
    transform_text=clean_text((input_sms))
#vectorize
    vector_input=tfidf.transform([transform_text])
#predict
    result=model.predict(vector_input)[0]
#Display
    if result==1:
        st.header("SPAM !!!")
    else:
        st.header("NOT SPAM :) ")