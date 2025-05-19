import streamlit as st
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Styling
st.set_page_config(
    page_title="Cancer AI Assistant", 
    page_icon="🧬", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced dark/light mode
st.markdown("""
    <style>
    :root {
        --primary: #2E86AB;
        --secondary: #A2D6F9;
        --background: #FFFFFF;
        --text: #333333;
        --accent: #FF6B6B;
        --gray: #6C757D;
        --success: #28A745;
    }

    /* Overall styling */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', system-ui, sans-serif;
        line-height: 1.6;
    }

    /* Main container */
    .main {
        background-color: var(--background);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Headers */
    .big-font {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid var(--secondary);
        padding-bottom: 0.5rem;
    }

    .subheader {
        font-size: 1.1rem !important;
        color: var(--gray) !important;
        margin-bottom: 2rem !important;
    }

    /* Input fields */
    .stTextArea textarea {
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 2px solid var(--secondary) !important;
    }

    /* Buttons */
    .stButton button {
        background: var(--primary) !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 134, 171, 0.4);
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    .card-title {
        font-size: 1.2rem;
        color: var(--gray);
        margin-bottom: 0.5rem;
    }

    .card-result {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 1rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        max-width: 80%;
    }

    .user-message {
        background: var(--primary);
        color: white;
        margin-left: auto;
    }

    .assistant-message {
        background: #f1f4f7;
        color: var(--text);
    }

    /* Animation */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary) 0%, #1b4965 100%);
        color: white;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        padding: 0.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.1);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Sidebar
with st.sidebar:
    st.title("🔍 Navigation")
    st.markdown("---")
    menu = st.radio("Select Feature:", 
                   ("🧬 Cancer Type Prediction", "💬 Cancer Chatbot"),
                   index=0)
    st.markdown("---")
    st.markdown("""
    <div style="color: var(--gray); font-size: 14px;">
        <p>This AI assistant helps with:</p>
        <ul>
            <li>Predicting cancer types from medical text</li>
            <li>Answering cancer-related questions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_prediction_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_prediction_model()
label_map = {0: 'Thyroid Cancer', 1: 'Colon Cancer', 2: 'Lung Cancer'}

# Stopwords
stop_words = stopwords.words('english')
for name in ['Colon_Cancer', 'Lung_Cancer', 'Thyroid_Cancer']:
    stop_words.append(name.lower())

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text_clean = " ".join([w.lower() for w in text.split() if w.lower() not in stop_words and len(w) > 1])
    tokens = word_tokenize(text_clean)
    return " ".join(tokens)

# Chatbot
@st.cache_data
def load_chatbot_dataset():
    df = pd.read_csv("qa_dataset_cancer.csv")
    return df

df_chatbot = load_chatbot_dataset()
chat_vectorizer = TfidfVectorizer()
tfidf_matrix = chat_vectorizer.fit_transform(df_chatbot['pertanyaan'])

def chatbot_response(user_input):
    user_tfidf = chat_vectorizer.transform([user_input])
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarity.argmax()
    confidence = similarity[0][index]
    if confidence > 0.3:
        return df_chatbot.iloc[index]['jawaban']
    else:
        return "I'm sorry, I don't understand your question. Could you please rephrase it or ask about something else related to cancer?"

# ==========================================
# 1. Cancer Prediction
# ==========================================
if "Prediction" in menu:
    st.markdown('<p class="big-font">🧬 Cancer Type Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Enter medical description or symptoms to predict cancer type</p>', unsafe_allow_html=True)

    user_input = st.text_area("✏️ Medical description:", height=150, 
                            placeholder="Enter symptoms, medical findings, or description here...")
    
    if st.button("🚀 Predict", key="predict_btn"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Analyzing the description..."):
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed])
                prediction = model.predict(vectorized)[0]
                
                st.markdown(f"""
                    <div class="prediction-card fade-in">
                        <div class="card-title">Prediction Result</div>
                        <div class="card-result">{label_map[prediction]}</div>
                        <p style='color: var(--gray);'>Based on the provided medical description</p>
                    </div>
                """, unsafe_allow_html=True)

# ==========================================
# 2. Cancer Chatbot
# ==========================================
elif "Chatbot" in menu:
    st.markdown('<p class="big-font">💬 Cancer Information Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Ask any questions about cancer (Thyroid, Colon, Lung)</p>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})