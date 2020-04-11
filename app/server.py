
#Streamlit (3rd part)
import streamlit as st
from fastai import *
from fastai.text import *

#Image View
from PIL import Image

#Import Document handlers (3rd Party)
import docx

## Suppress warnings
import warnings
warnings.filterwarnings('ignore')

st.title("Legal Clause Classifier")

st.write("Using this classifier you can see how our model behaves as well as test any part of your contract to see the outputs.")

path = Path(__file__).parent
image = Image.open(path/"images/Confusion_matrix.png")
st.image(image)

text = st.text_area("Enter your text here:")

if text:
    st.write("Your Input text:", text)

st.header("Alternatively, you can upload a text file.")





## Upload a text file
if st.checkbox("Upload a text file"):
    file = st.file_uploader("Upload a text file:",type=['txt'])
    if file:
        text = file.read()
        st.write(text)

## upload a docx file
def getParaText(doc):
    fullText = [para.text for para in doc.paragraphs if len(para.text)>0]
    #return '\nNew Paragraph\n'.join(fullText)
    return fullText

if st.checkbox("Upload a docx file"):
    file = st.file_uploader("Upload a text file:",type=['docx'])
    def process_docx(file):
        doc = docx.Document(file)
        para_text = getParaText(doc)

        ## Display Full Text
        if st.checkbox("Show Full Document"):
            st.write(para_text)
            
        max_len = len(para_text)
        index = st.slider(label='Paragraph Number', min_value = 0, max_value=max_len, value=10)
        text  = para_text[index]
        st.write({'Paragraph No.': index, 'Paragraph Text':text})
        return text
    if file:
        text = process_docx(file)
    




#@st.cache
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
    #st.write("Analyzed Text:", text)

