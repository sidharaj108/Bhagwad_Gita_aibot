import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

st.set_page_config(page_title="🕉️ Bhagavad Gita AI", page_icon="🕉️", layout="centered")

st.title("🕉️ Bhagavad Gita AI Chatbot")
st.markdown("**English • हिंदी • ગુજરાતી**")

# Load Data
@st.cache_data(show_spinner="Loading verses...")
def load_data():
    df = pd.read_excel("Bhagwad_Gita_contant.xlsx")
    df['combined_meaning'] = df['HinMeaning'].fillna('') + " " + df['EngMeaning'].fillna('') + " " + df.get('WordMeaning', '').fillna('')
    df['search_text'] = df['combined_meaning'].str.lower().str.strip()
    return df

df = load_data()

# Load Model
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()
corpus_embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)

def clean_word_meaning(text):
    if not isinstance(text, str):
        return "", ""
    if "Commentary" in text:
        parts = text.split("Commentary", 1)
        word_mean = parts[0].strip()
        comm = "Commentary" + parts[1].strip()
    else:
        word_mean = text.strip()
        comm = ""
    return re.sub(r'\s+', ' ', word_mean).strip(), comm

def gita_chatbot(query: str):
    if not query or query.strip() == "":
        return "🙏 Please ask something or enter verse like 1.4"

    query = query.strip()

    # Direct Verse
    if re.match(r'^\d+\.\d+$', query):
        ch, vs = map(int, query.split('.'))
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

    # Semantic Search - More Forgiving
    try:
        lang = detect(query.lower())
    except:
        lang = "en"

    is_guj = any(w in query.lower() for w in ["gujarati", "gujrati", "guj", "ગુજરાતી"])
    output_lang = 'gu' if is_guj or lang == 'gu' else 'en'

    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=12)[0]

    results = []
    for hit in hits:
        if hit['score'] < 0.22:   # Very forgiving threshold
            continue
        row = df.iloc[hit['corpus_id']]
        word_mean, comm = clean_word_meaning(row.get('WordMeaning', ''))
        meaning = row.get('gujarati_meaning', row['HinMeaningEngMeaning']) if output_lang == 'gu' else row['HinMeaningEngMeaning']

        results.append(f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}** (Score: {hit['score']:.2f})

**श्लोक**  
{row['Shloka']}

**Word Meaning**  
{word_mean}

**Meaning**  
{meaning}
---
""")

    if results:
        return "\n".join(results)
    else:
        return """**No strong match found.**

**Better questions to try:**
- Karma Yoga
- What is Karma Yoga?
- Karm yog kya hai?
- Duty without attachment
- Chapter 3
- 3.1
- 18.66
- કર્મયોગ શું છે?
"""

# ====================== UI ======================
query = st.text_input(
    "Ask your question",
    placeholder="What is Karma Yoga?   or   18.66",
    key="query"
)

if st.button("Get Wisdom 🕉️", type="primary"):
    with st.spinner("Finding wisdom from Bhagavad Gita..."):
        response = gita_chatbot(query)
        st.markdown(response)

# Sidebar
st.sidebar.header("Quick Examples")
examples = [
    "What is Karma Yoga?",
    "Karma Yoga",
    "What is the meaning of duty?",
    "Explain Dharma",
    "1.4",
    "2.22",
    "18.66",
    "કર્મયોગ શું છે?"
]

for ex in examples:
    if st.sidebar.button(ex):
        st.session_state.query = ex
        st.rerun()

st.sidebar.info("Tip: Lower score threshold helps find more verses.")
