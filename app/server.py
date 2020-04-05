
#Streamlit (3rd part)
import streamlit as st
from fastai import *
from fastai.text import *

#Image View
from PIL import Image

st.title("Legal Clause Classifier")

st.write("Using this classifier you can see how our model behaves as well as test any part of your contract to see the outputs.")

path = Path(__file__).parent
image = Image.open(path/"images/Confusion_matrix.png")
st.image(image)

text = st.text_area("Enter your text here:")

if text:
    st.write("Your Input text:", text)

#st.header("Alternatively, you can upload a text file.")

#file = st.file_uploader("Upload a text file:",type=['txt'])

#if file:
#    df = pd.read_csv(file)
#    st.write(df)


def setup_learner():
    learn_fwd = load_learner(path/'models','model_exported.pkl')
    return learn_fwd

 
def analyze(learn, text):
    class_, ind, probs = learn.predict(text)
    score = probs[ind].item()
    if score < 0.5:
        confidence = 'Low'
    else:
        confidence = 'High'
    return class_, score, confidence


if st.button("Analyze"):
    st.write("Analyzing...")
    learn = setup_learner()
    class_, score, confidence = analyze(learn, text)
    st.write("Predicted Label:", class_)
    st.write("Predicted Score:", score)
    st.write("Confidence:", confidence)
