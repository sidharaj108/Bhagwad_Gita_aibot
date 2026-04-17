import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# Page Configuration
st.set_page_config(
    page_title="🕉️ Bhagavad Gita AI",
    page_icon="🕉️",
    layout="centered"
)

st.title("🕉️ Bhagavad Gita AI Chatbot")
st.markdown("**Multi-language Support**: English • हिंदी • ગુજરાતી")

# Load Data
@st.cache_data(show_spinner="Loading Bhagavad Gita verses...")
def load_data():
    df = pd.read_excel("Bhagwad_Gita_contant.xlsx")
    df['HinMeaningEngMeaning'] = df['HinMeaning'].fillna('') + " " + df['EngMeaning'].fillna('')
    df['search_text'] = df['HinMeaningEngMeaning'].str.lower().str.strip()
    return df

df = load_data()

# Load Model
@st.cache_resource(show_spinner="Loading AI model (first time may take 30-60 seconds)...")
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()
corpus_embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)

# Helper Function
def clean_word_meaning(text):
    if not isinstance(text, str):
        return "", ""
    if "Commentary" in text:
        parts = text.split("Commentary", 1)
        word_meaning = parts[0].strip()
        commentary = "Commentary" + parts[1].strip()
    else:
        word_meaning = text.strip()
        commentary = ""
    word_meaning = re.sub(r'\s+', ' ', word_meaning).strip()
    return word_meaning, commentary

# Chatbot Function
def gita_chatbot(query: str):
    if not query or query.strip() == "":
        return "🙏 Please ask a question or enter verse like **1.4**"

    query = query.strip()

    # Direct Verse Search
    verse_match = re.match(r'^(\d+)\.(\d+)$', query)
    if verse_match:
        ch = int(verse_match.group(1))
        vs = int(verse_match.group(2))
        res = df[(df['Chapter'] == ch) & (df['Verse'] == vs)]
        if not res.empty:
            row = res.iloc[0]
            word_mean, comm = clean_word_meaning(row.get('WordMeaning', ''))
            meaning = row.get('gujarati_meaning', row['HinMeaningEngMeaning'])
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

    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]

    results = []
    for hit in hits:
        if hit['score'] < 0.38:
            continue
        row = df.iloc[hit['corpus_id']]
        word_mean, comm = clean_word_meaning(row.get('WordMeaning', ''))
        final_meaning = row.get('gujarati_meaning', row['HinMeaningEngMeaning']) if output_lang == 'gu' else row['HinMeaningEngMeaning']
        
        results.append(f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}**

**श्लोक**  
{row['Shloka']}

**Word Meaning**  
{word_mean}

**Meaning**  
{final_meaning}
---
""")

    return "\n".join(results) if results else "❌ No relevant verse found. Try different words or enter verse like 2.22"

# UI
query = st.text_input(
    "🙏 Ask anything or enter verse number",
    placeholder="What is the meaning of duty? in gujarati    or    18.66"
)

if st.button("Get Wisdom 🕉️", type="primary"):
    with st.spinner("Finding divine wisdom from Shrimad Bhagavad Gita..."):
        response = gita_chatbot(query)
        st.markdown(response)

# Sidebar Examples
st.sidebar.header("🔥 Quick Examples")
examples = [
    "What is Karma Yoga?",
    "Explain Dharma",
    "What is the meaning of duty? in gujarati",
    "1.4",
    "2.22",
    "18.66",
    "કર્તવ્યનો અર્થ શું છે?"
]

for ex in examples:
    if st.sidebar.button(ex):
        st.session_state.query = ex   # This may not work perfectly, but it's fine
        st.rerun()
