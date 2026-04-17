import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🕉️ Bhagavad Gita AI Chatbot",
    page_icon="🕉️",
    layout="centered"
)

st.title("🕉️ Bhagavad Gita AI Chatbot")
st.markdown("**English • हिंदी • ગુજરાતી**")

# ====================== LOAD DATA ======================
@st.cache_data(show_spinner="Loading Bhagavad Gita verses...")
def load_data():
    df = pd.read_excel("Bhagwad_Gita_contant.xlsx")
    
    # Safely create combined meaning column
    if 'HinMeaning' not in df.columns:
        df['HinMeaning'] = ""
    if 'EngMeaning' not in df.columns:
        df['EngMeaning'] = ""
    if 'gujarati_meaning' not in df.columns:
        df['gujarati_meaning'] = ""

    df['combined_meaning'] = (
        df['HinMeaning'].fillna('') + " " + 
        df['EngMeaning'].fillna('') + " " + 
        df['gujarati_meaning'].fillna('')
    )
    df['search_text'] = df['combined_meaning'].str.lower().str.strip()
    
    print(f"✅ Loaded {len(df)} verses. Columns: {list(df.columns)}")
    return df

df = load_data()

# ====================== LOAD MODEL ======================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()
corpus_embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)

# ====================== HELPER FUNCTION ======================
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

# ====================== CHATBOT FUNCTION ======================
def gita_chatbot(query: str):
    if not query or query.strip() == "":
        return "🙏 Please ask a question or enter verse like **1.4**"

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
            # Safe meaning selection
            if 'gujarati_meaning' in row and pd.notna(row['gujarati_meaning']) and str(row['gujarati_meaning']).strip():
                meaning = row['gujarati_meaning']
            else:
                meaning = row.get('HinMeaning', '') + " " + row.get('EngMeaning', '')
            return f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}**

**श्लोक**  
{row.get('Shloka', 'Not available')}

**Word Meaning**  
{word_mean}

**Commentary**  
{comm if comm else "No commentary available."}

**Meaning**  
{meaning.strip()}
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
        if hit['score'] < 0.22:
            continue
            
        row = df.iloc[hit['corpus_id']]
        word_mean, comm = clean_word_meaning(row.get('WordMeaning', ''))
        
        # Safe meaning selection
        if output_lang == 'gu' and 'gujarati_meaning' in df.columns:
            meaning = row.get('gujarati_meaning', '')
        else:
            meaning = row.get('HinMeaning', '') + "\n\n" + row.get('EngMeaning', '')
        
        if not meaning.strip():
            meaning = row.get('combined_meaning', 'Meaning not available')

        results.append(f"""
**Chapter {row['Chapter']}, Verse {row['Verse']}** (Relevance: {hit['score']:.2f})

**श्लोक**  
{row.get('Shloka', 'Not available')}

**Word Meaning**  
{word_mean}

**Meaning**  
{meaning.strip()}
---
""")

    if results:
        return "\n".join(results)
    else:
        return f"""❌ No relevant verse found for **"{query}"**.

**Try these questions:**
- What is Karma Yoga?
- Karm yog kya hai?
- What is duty?
- 1.4
- 18.66
- કર્મયોગ શું છે?
"""

# ====================== UI ======================
query = st.text_input(
    "🙏 Ask your question about Bhagavad Gita",
    placeholder="What is Karma Yoga?   or   18.66   or   કર્મયોગ શું છે?",
    key="user_query"
)

if st.button("Get Wisdom 🕉️", type="primary"):
    with st.spinner("Finding wisdom from Shrimad Bhagavad Gita..."):
        response = gita_chatbot(query)
        st.markdown(response)

# Sidebar Examples
st.sidebar.header("🔥 Quick Examples")
examples = [
    "What is Karma Yoga?",
    "What is the meaning of duty?",
    "Explain Dharma",
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
st.sidebar.info("💡 Tip: Try typing 'Karma Yoga' or click the examples.")
