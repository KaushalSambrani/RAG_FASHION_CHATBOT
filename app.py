import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment
load_dotenv()

# Load embedding model and FAISS DB (cached)
@st.cache_resource
def load_embeddings_and_db():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db

db = load_embeddings_and_db()

# Optional price filter function
def filter_docs_by_price(docs, max_price):
    filtered = []
    for doc in docs:
        lines = doc.page_content.lower().split("\n")
        for line in lines:
            if "price" in line:
                digits = ''.join(filter(str.isdigit, line))
                try:
                    price = int(digits)
                    if price <= max_price:
                        filtered.append(doc)
                except:
                    pass
    return filtered if filtered else docs  # fallback

# Set up Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.4,
)
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
You are Taarini Threads' digital assistant. Use ONLY the context below to answer user questions. DO NOT use any outside knowledge or guess—if the context does not contain the answer, state that honestly.

If the user asks for policies or brand rules, explicitly mention which policy document your answer is based on (e.g., 'From: shipping.txt').

Display Products in a markdown format for example: 
### SKU and Product Name -- Price 
- Size
- Ratings                                         

If you do not find enough products matching the user's criteria, say so and list what you have.

Always conclude with:
"Check out our website and follow us on Instagram: https://www.instagram.com/taarini_threads/ for the latest updates!"

Context:
{context}

Question:
{question}

Answer:
""")

# Streamlit UI config
st.set_page_config(page_title="🧵 Taarini Threads Assistant", layout="wide")
st.title("🧵 Taarini Threads Assistant")

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("Ask me anything about products, return policy, etc.")

if query:
    with st.spinner("Thinking..."):
        # Check if price filter is needed
        price_match = re.search(r'under\s*rs\s*(\d+)', query.lower())
        max_price = int(price_match.group(1)) if price_match else None

        # Retrieve docs
        retrieved_docs = db.similarity_search(query, k=7)
        if max_price:
            retrieved_docs = filter_docs_by_price(retrieved_docs, max_price)
        # Extract numeric rating from doc text
        def extract_rating(doc):
            match = re.search(r'ratings:\s*([\d.]+)', doc.page_content.lower())
            return float(match.group(1)) if match else 0.0

# Sort by rating (descending)
        retrieved_docs.sort(key=extract_rating, reverse=True)
        # Format context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Prompt LLM
        full_prompt = custom_prompt.format(context=context, question=query)
        response = llm.invoke(full_prompt)
        response_text = response.content



        # Store chat
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", response_text))

# Display chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
