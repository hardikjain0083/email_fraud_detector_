import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import pytesseract
import io
import os
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Phishing Detector | Team Pyrates",
    page_icon="🏴‍☠️",
    layout="wide"
)

# --- Global Styling ---
# Injecting custom CSS for a consistent, cool theme across all pages.
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    /* Main app background and text color */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Poppins', sans-serif;
    }

    /* --- General Component Styling --- */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif; /* Techy font for headers */
    }

    /* Styling for all buttons */
    .stButton>button {
        border: 1px solid #30363d;
        border-radius: 20px;
        color: #c9d1d9;
        background-color: #21262d;
        transition: all 0.2s ease-in-out;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.5);
    }
    
    /* Styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #161b22; /* Slightly lighter than main bg */
		border-radius: 8px 8px 0px 0px;
		gap: 1px;
		padding: 10px 20px;
        border: 1px solid #30363d;
        border-bottom: none;
        font-family: 'Poppins', sans-serif;
    }
    .stTabs [aria-selected="true"] {
  		background-color: #21262d;
        border-bottom: 2px solid #58a6ff;
        color: #58a6ff;
	}

    /* Style for text area and file uploader */
    .stTextArea textarea, .stFileUploader {
        border: 1px solid #30363d;
        background-color: #161b22;
        border-radius: 8px;
        color: #c9d1d9;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.5);
    }
    
    /* Custom styling for result boxes */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border-width: 1px;
    }

</style>
""", unsafe_allow_html=True)


# --- NLTK Data Download (Cached) ---
@st.cache_resource
def download_nltk_data():
    packages = ['punkt', 'stopwords', 'wordnet']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package)
    return True

download_nltk_data()

# --- Load Model (Cached) ---
@st.cache_resource
def load_pipeline():
    model_path = 'phishing_detector_pipeline.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: '{model_path}'. Please ensure it's in the same directory.")
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

loaded_pipeline = load_pipeline()

# --- Helper Functions ---
def preprocess_text(text):
    """
    Cleans and prepares a single text entry.
    NLTK objects are instantiated here for better stability with Streamlit.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Handle potential non-string inputs gracefully
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

def create_meta_features(df):
    df['url_count'] = df['Full_Text'].apply(lambda x: len(re.findall(r'http\S+|www\S+|https\S+', str(x))))
    df['text_length'] = df['Full_Text'].apply(lambda x: len(str(x)))
    df['special_char_count'] = df['Full_Text'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', str(x))))
    df['digit_count'] = df['Full_Text'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['uppercase_word_count'] = df['Full_Text'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', str(x))))
    return df

# --- Page Rendering Functions ---

def render_home_page():
    """Renders the stylish title page."""
    st.markdown("""
    <style>
        .main-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 90vh;
            background: linear-gradient(rgba(13, 17, 23, 0.95), rgba(13, 17, 23, 0.95)), url(https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=2070&auto=format&fit=crop);
            background-size: cover;
            background-position: center;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            border: 1px solid #30363d;
        }
        .title {
            font-family: 'Orbitron', sans-serif;
            font-size: 5.5rem;
            color: #58a6ff;
            text-shadow: 0 0 15px rgba(88, 166, 255, 0.7);
        }
        .team-name {
            font-family: 'Poppins', sans-serif;
            font-weight: 300;
            font-size: 1.5rem;
            color: #8b949e;
            margin-top: -1.5rem;
        }
        .team-members {
            margin-top: 3rem;
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        .member-card {
            background-color: #161b22;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            border: 1px solid #30363d;
            font-family: 'Poppins', sans-serif;
            font-size: 1.1rem;
            font-weight: 400;
            color: #c9d1d9;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .member-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(88, 166, 255, 0.3);
            border-color: #58a6ff;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-container">
        <div class="title">Phishing Detector</div>
        <div class="team-name">A Project by Team Pyrates 🏴‍☠️</div>
        <div class="team-members">
            <div class="member-card">Hardik Jain</div>
            <div class="member-card">Divyaraj Rajpurohit</div>
            <div class="member-card">Mishita Tiwari</div>
            <div class="member-card">Vinita Sharma</div>
            <div class="member-card">Priyamvada Tiwari</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer
    cols = st.columns([3, 2, 3])
    with cols[1]:
        if st.button("🚀 Launch Detector", use_container_width=True):
            st.session_state.page = 'detector'
            st.rerun()

def render_detector_page():
    """Renders the main phishing detector application."""
    st.title("Phishing Email Detector 🎣")
    st.markdown("This tool analyzes email content to detect potential phishing attempts. Enter text or upload a file below.")
    st.write("---")

    def classify_and_display(text_content):
        if not text_content or not text_content.strip():
            st.warning("Please provide some text to analyze.")
            return

        word_count = len(text_content.split())
        has_url = re.search(r'http\S+|www\S+|https\S+', text_content)
        if word_count < 7 and not has_url:
            st.success("**Result: LEGITIMATE** ✅")
            st.info("_(Determined by heuristic rule for very short text.)_")
            return

        input_df = pd.DataFrame([text_content], columns=['Full_Text'])
        input_df = create_meta_features(input_df)
        input_df['processed_text'] = input_df['Full_Text'].apply(preprocess_text)

        prediction = loaded_pipeline.predict(input_df)[0]
        prediction_proba = loaded_pipeline.predict_proba(input_df)[0]
        confidence = max(prediction_proba)
        CONFIDENCE_THRESHOLD = 0.80

        st.subheader("Analysis Result")
        if confidence < CONFIDENCE_THRESHOLD:
            leaning_towards = "PHISHING 🎣" if prediction == 1 else "LEGITIMATE ✅"
            st.info(f"**Result: UNCERTAIN** 🤔")
            st.write(f"The model is leaning towards **{leaning_towards}**, but its confidence ({confidence:.2%}) is below the required {CONFIDENCE_THRESHOLD:.0%} threshold.")
        elif prediction == 1:
            st.warning(f"**Result: PHISHING** 🎣 (Confidence: {confidence:.2%})")
        else:
            st.success(f"**Result: LEGITIMATE** ✅ (Confidence: {confidence:.2%})")

    tab1, tab2 = st.tabs(["Paste Text", "Upload File (TXT or Image)"])

    with tab1:
        text_input = st.text_area("Enter the full email text here:", height=250, placeholder="Paste email content...")
        if st.button("Classify Pasted Text", type="primary"):
            if loaded_pipeline:
                classify_and_display(text_input)
            else:
                st.error("Model is not loaded. Cannot perform classification.")

    with tab2:
        uploaded_file = st.file_uploader("Upload an email (.txt) or a screenshot (.png, .jpg)", type=['txt', 'png', 'jpg', 'jpeg'])
        if uploaded_file is not None and loaded_pipeline:
            content_to_classify = ""
            try:
                if "text" in uploaded_file.type:
                    content_to_classify = io.BytesIO(uploaded_file.getvalue()).read().decode('utf-8')
                elif "image" in uploaded_file.type:
                    image = Image.open(uploaded_file)
                    content_to_classify = pytesseract.image_to_string(image)
                
                classify_and_display(content_to_classify)
                
                if content_to_classify:
                    with st.expander("View Extracted Text"):
                        st.text(content_to_classify)
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # Back button placed in a column for better layout
    st.write("---")
    cols_back = st.columns([5, 1, 5])
    with cols_back[1]:
        if st.button("⬅️ "):
            st.session_state.page = 'home'
            st.rerun()

# --- Main App Router ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    render_home_page()
else:
    render_detector_page()
