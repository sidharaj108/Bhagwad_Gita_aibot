import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🕉️ Bhagavad Gita AI Chatbot",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🕉️ Bhagavad Gita AI Chatbot")
st.markdown("**Multi-language Support**: English • हिंदी • ગુજરાતી")

# ====================== LOAD DATA ======================
@st.cache_data(show_spinner="Loading sacred verses...")
def load_data():
    df = pd.read_excel("Bhagwad_Gita_contant.xlsx")
    df['HinMeaningEngMeaning'] = df['HinMeaning'].fillna('') + " " + df['EngMeaning'].fillna('')
    df['search_text'] = df['HinMeaningEngMeaning'].str.lower().str.strip()
    return df

df = load_data()

# ====================== LOAD MODEL ======================
@st.cache_resource(show_spinner="Loading AI model... (first load takes 30-60 sec)")
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()
corpus_embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)

# ====================== HELPER FUNCTIONS ======================
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

# ====================== MAIN CHATBOT FUNCTION ======================
def gita_chatbot(query: str):
    if not query or query.strip() == "":
        return "🙏 Please ask a question or enter a verse like **1.4** or **18.66**."

    query = query.strip()

    # Direct Verse Lookup
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
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=10)[0]

    results = []
    for hit in hits:
        if hit['score'] < 0.28:
            continue
        row = df.iloc[hit['corpus_id']]
        word_mean, comm = clean_word_meaning(row.get('WordMeaning', ''))
        final_meaning = row.get('gujarati_meaning', row['HinMeaningEngMeaning']) if output_lang == 'gu' else row['HinMeaningEngMeaning']

        results.append(f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}** (Relevance: {hit['score']:.2f})

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

    if results:
        return "\n".join(results)
    else:
        return f"""❌ Sorry, I couldn't find a strong match for **"{query}"**.

**Try these:**
- "What is Karma Yoga?"
- "Karm yog kya hai?"
- "What is the meaning of duty?"
- "3.1" or "18.66"
"""

# ====================== STREAMLIT UI ======================
# Initialize session state for query
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

query = st.text_input(
    "🙏 Ask any question about Bhagavad Gita",
    value=st.session_state.user_query,
    placeholder="What is Karma Yoga?   or   18.66   or   કર્મયોગ શું છે?",
    key="query_input"
)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("Get Wisdom 🕉️", type="primary"):
        with st.spinner("Seeking divine wisdom from Shrimad Bhagavad Gita..."):
            response = gita_chatbot(query)
            st.markdown(response)

with col2:
    if st.button("Clear Input"):
        st.session_state.user_query = ""
        st.rerun()

# ====================== SIDEBAR ======================
st.sidebar.header("🔥 Quick Examples")

examples = [
    "What is Karma Yoga?",
    "What is the meaning of duty?",
    "Explain Dharma",
    "Who is a true yogi?",
    "1.4",
    "2.22",
    "18.66",
    "કર્મયોગ શું છે?",
    "कर्तव्य का क्या अर्थ है?"
]

for ex in examples:
    if st.sidebar.button(ex, key=f"btn_{ex}"):
        st.session_state.user_query = ex
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Ask in English, Hindi, or Gujarati. Direct verse numbers (like 3.1) work best.")
