import sys
from pathlib import Path

# Add project root directory to sys.path so app.py can import from train/
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the text preprocessing function from the file where your friend trained the model
from train.lr_classifier import preprocesses_text  # Or svm_classifier depending on which was used

import streamlit as st
import pickle

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    load_css(css_path)

st.markdown(
            """
            <div class="title-text">AI-Written Essay Classifier
            </div>
            """,
            unsafe_allow_html=True,
            )

st.markdown(
            """
            <div class="description">In about 50 words, tell us about the best TV show you have ever watch,and why? Explain a little, like who's you favorite character, or favorite scenes, and etc.: \n
            </div>
            """,
            unsafe_allow_html=True,
            )

try: 
    with open('model/lr_model.pkl', 'rb') as file:
        pipeline = pickle.load(file)

except FileNotFoundError:
    st.error("Model file not found. Please ensure 'lr_model.pkl' exists in the 'model' directory.")
    st.stop()

user_text = st.text_area("Enter here:", height=250)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.markdown('<div class="result-box warning-result">Please paste some text first.</div>', unsafe_allow_html=True)
    else:
        prediction = pipeline.predict([user_text])

        if prediction[0] == 1:
            st.markdown('<div class="result-box ai-result">This is likely AI-Generated.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box human-result">This is likely Human-Written.</div>', unsafe_allow_html=True)