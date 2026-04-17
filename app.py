# app.py - Bhagavad Gita AI Chatbot for Streamlit

import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="🕉️ Bhagavad Gita AI",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🕉️ Bhagavad Gita AI Chatbot")
st.markdown("**Multi-language Support**: English • हिंदी • ગુજરાતી")

# --------------------- Load Data ---------------------
@st.cache_data(show_spinner="Loading Bhagavad Gita verses...")
def load_data():
    df = pd.read_excel("Bhagwad_Gita_contant.xlsx")
    df['HinMeaningEngMeaning'] = df['HinMeaning'].fillna('') + " " + df['EngMeaning'].fillna('')
    df['search_text'] = df['HinMeaningEngMeaning'].str.lower().str.strip()
    df['verse_key'] = df['Chapter'].astype(str) + '.' + df['Verse'].astype(str)
    return df

df = load_data()

# --------------------- Load Model ---------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()
corpus_embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)

# --------------------- Helper Functions ---------------------
def clean_word_meaning(text):
    if not isinstance(text, str):
        return text, ""
    if "Commentary" in text:
        parts = text.split("Commentary", 1)
        word_meaning = parts[0].strip()
        commentary = "Commentary" + parts[1].strip()
    else:
        word_meaning = text.strip()
        commentary = ""
    word_meaning = re.sub(r'\s+', ' ', word_meaning).strip()
    return word_meaning, commentary

# --------------------- Chatbot Logic ---------------------
def gita_chatbot(query: str):
    if not query or query.strip() == "":
        return "🙏 Please ask a question or enter a verse like **1.4**"

    query = query.strip()

    # Direct Verse Search
    verse_match = re.match(r'^(\d+)\.(\d+)$', query)
    if verse_match:
        ch, vs = int(verse_match.group(1)), int(verse_match.group(2))
        res = df[(df['Chapter'] == ch) & (df['Verse'] == vs)]
        if not res.empty:
            row = res.iloc[0]
            word_mean, comm = clean_word_meaning(row['WordMeaning'])
            meaning = row['gujarati_meaning'] if 'gu' in query.lower() else row['HinMeaningEngMeaning']
            return f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}**

**श्लोक**  
{row['Shloka']}

**Word Meaning**  
{word_mean}

**Commentary**  
{comm if comm else "No commentary available."}

**Meaning**  
{meaning}
"""

    # Semantic Search
    try:
        detected_lang = detect(query.lower())
    except:
        detected_lang = "en"

    user_guj = any(w in query.lower() for w in ["gujarati", "gujrati", "guj", "ગુજરાતી"])
    output_lang = 'gu' if user_guj or detected_lang == 'gu' else 'en'

    query_clean = query.lower().strip() if output_lang == 'en' else query.strip()
    query_emb = model.encode(query_clean, convert_to_tensor=True)

    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]

    results = []
    for hit in hits:
        if hit['score'] < 0.38:
            continue
        row = df.iloc[hit['corpus_id']]
        word_mean, comm = clean_word_meaning(row['WordMeaning'])
        final_meaning = row['gujarati_meaning'] if output_lang == 'gu' else row['HinMeaningEngMeaning']

        results.append(f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}**

**श्लोक**  
{row['Shloka']}

**Word Meaning**  
{word_mean}

**Commentary**  
{comm if comm else ""}

**Meaning**  
{final_meaning}
---
""")

    return "\n".join(results) if results else "❌ No relevant verse found. Try different words or enter a verse number (e.g., 2.22)."

# --------------------- Streamlit UI ---------------------
query = st.text_input(
    "🙏 Ask anything about Bhagavad Gita (English / Hindi / Gujarati)",
    placeholder="What is the meaning of duty? in gujarati   or   18.66",
    key="query_input"
)

if st.button("Get Wisdom 🕉️", type="primary"):
    with st.spinner("Thinking..."):
        response = gita_chatbot(query)
        st.markdown(response)

# Sidebar with Examples
st.sidebar.header("Quick Examples")
examples = [
    "What is Karma Yoga?",
    "Explain the concept of Dharma",
    "What is the meaning of duty? in gujarati",
    "1.4",
    "18.66",
    "Give summary of Chapter 2",
    "કર્તવ્યનો અર્થ શું છે?",
    "कर्तव्य का क्या अर्थ है?"
]

for ex in examples:
    if st.sidebar.button(ex):
        st.session_state.query_input = ex
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Tip: You can directly type verse numbers like **2.22** or ask in Gujarati/Hindi.")
