import streamlit as st
import torch
import torch.nn.functional as F
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ZENDS AI Support System",
    layout="wide",
    page_icon="📡"
)

# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1558494949-ef010cbdcc31");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay */
.stApp:before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(10, 25, 47, 0.85);
    z-index: -1;
}

/* Glass container */
.block-container {
    background: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-size: 18px;
    font-weight: bold;
}

/* Metric styling */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

/* Text color */
h1, h2, h3, h4, h5, h6, p, span {
    color: white !important;
}

/* Chat message styling */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📡 ZENDS Telecom Intelligent Support System")

# =========================================================
# SESSION STATE INIT
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "user", "message": "My internet connection is very slow."},
        {"role": "assistant", "message": "Please restart your router and ensure cables are connected properly."}
    ]

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "total_queries": 1,
        "intent_counts": {},
        "sentiment_counts": {}
    }

# =========================================================
# LABELS
# =========================================================
INTENT_LABELS = {
    0: "Technical",
    1: "Billing",
    2: "Complaint",
    3: "Product Inquiry",
    4: "Refund"
}

SENTIMENT_LABELS = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    intent_tokenizer = AutoTokenizer.from_pretrained("intent_model")
    intent_model = AutoModelForSequenceClassification.from_pretrained("intent_model")

    sentiment_tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    return (
        intent_model, intent_tokenizer,
        sentiment_model, sentiment_tokenizer,
        embedding_model,
        llm_model, llm_tokenizer
    )

(
    intent_model, intent_tokenizer,
    sentiment_model, sentiment_tokenizer,
    embedding_model,
    llm_model, llm_tokenizer
) = load_models()

# =========================================================
# VECTOR DB
# =========================================================
@st.cache_resource
def load_vector_db():
    client = chromadb.Client(
        Settings(
            persist_directory="chroma_db",
            anonymized_telemetry=False
        )
    )
    return client.get_or_create_collection("zends_collection")

collection = load_vector_db()

# =========================================================
# FUNCTIONS
# =========================================================
def predict_intent(query):
    inputs = intent_tokenizer(query, return_tensors="pt", truncation=True)
    outputs = intent_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    pred = torch.argmax(probs).item()
    return INTENT_LABELS.get(pred), round(confidence * 100, 2)

def predict_sentiment(query):
    inputs = sentiment_tokenizer(query, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    pred = torch.argmax(probs).item()
    return SENTIMENT_LABELS.get(pred), round(confidence * 100, 2)

def retrieve_context(query):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return "\n".join(results["documents"][0])

def generate_response(query):
    context = retrieve_context(query)
    prompt = f"""
You are a professional telecom support assistant.
Answer clearly using ONLY the context.

Context:
{context}

Question:
{query}
"""
    inputs = llm_tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["👤 Customer Chat", "🎧 Support Agent", "📊 Analytics Dashboard"]
)

# =========================================================
# CUSTOMER CHAT
# =========================================================
with tab1:

    st.subheader("Customer Conversation")

    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).write(chat["message"])

    user_query = st.chat_input("Type your question here...")

    if user_query:
        intent, _ = predict_intent(user_query)
        sentiment, _ = predict_sentiment(user_query)

        response = generate_response(user_query)

        st.session_state.chat_history.append(
            {"role": "user", "message": user_query}
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "message": response}
        )

        st.session_state.analytics["total_queries"] += 1
        st.session_state.analytics["intent_counts"][intent] = \
            st.session_state.analytics["intent_counts"].get(intent, 0) + 1
        st.session_state.analytics["sentiment_counts"][sentiment] = \
            st.session_state.analytics["sentiment_counts"].get(sentiment, 0) + 1

        st.rerun()

# =========================================================
# SUPPORT AGENT (EDIT ANY PREVIOUS)
# =========================================================
with tab2:

    st.subheader("Agent Review Panel")

    assistant_indices = [
        i for i, msg in enumerate(st.session_state.chat_history)
        if msg["role"] == "assistant"
    ]

    if assistant_indices:

        options = {
            f"Reply #{idx//2 + 1}": idx
            for idx in assistant_indices
        }

        selected_label = st.selectbox(
            "Select Reply to Review / Edit",
            list(options.keys())
        )

        selected_index = options[selected_label]

        customer_query = st.session_state.chat_history[selected_index - 1]["message"]

        st.markdown("### Related Customer Message")
        st.write(customer_query)

        intent, intent_conf = predict_intent(customer_query)
        sentiment, sent_conf = predict_sentiment(customer_query)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Intent", intent)
            st.metric("Confidence", f"{intent_conf}%")

        with col2:
            st.metric("Sentiment", sentiment)
            st.metric("Confidence", f"{sent_conf}%")

        edited_response = st.text_area(
            "Edit This Reply",
            value=st.session_state.chat_history[selected_index]["message"],
            height=200
        )

        if st.button("Update Selected Reply"):
            st.session_state.chat_history[selected_index]["message"] = edited_response
            st.success("Reply updated instantly!")
            st.rerun()

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================
with tab3:

    st.subheader("System Analytics Dashboard")

    st.metric("Total Queries",
              st.session_state.analytics["total_queries"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Intent Distribution")
        if st.session_state.analytics["intent_counts"]:
            st.bar_chart(st.session_state.analytics["intent_counts"])

    with col2:
        st.markdown("### Sentiment Distribution")
        if st.session_state.analytics["sentiment_counts"]:
            st.bar_chart(st.session_state.analytics["sentiment_counts"])
