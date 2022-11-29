import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from joblib import parallel, delayed
import joblib

import streamlit as st

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


model = joblib.load('model.pkl')

def predict(text):
    txt = text
    txt = stem_words(txt)
    txt = lemmatize_words(txt)
    txt = [txt]
    if model.predict(txt)== 1:
        return 'True'
    else:
        return 'Fake'


def main():
    html_temp = """
    <div class="mainhd">
    <div class="chd"><h2 class="hd">  Fake News Detection </h2></div>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    review = st.text_input("","Type Here")
    safe_html="""  
        <div class="ifnt">
        <p class="ifntt"> Fake News.</p  >
        </div>
    """
    danger_html="""  
        <div clas="ift">
        <p class="iftt"> Genuine News. </p>
        </div>
    """

    if st.button("Predict"):
        output=predict(review)
        st.success('The news is {}'.format(output))

        if output == 'True':
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

    

if __name__=='__main__':
    main()
