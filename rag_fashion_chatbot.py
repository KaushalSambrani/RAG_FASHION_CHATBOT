# rag_fashion_chatbot.py

import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Load product JSON
def load_product_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    docs = [
        Document(
            page_content="\n".join([f"{k}: {v}" for k, v in p.items()]),
            metadata={"source": "product_catalog"}
        )
        for p in products
    ]
    return docs

# Load text policy files
def load_text_documents(folder_path):
    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": fname}))
    return docs

# Split all docs into chunks
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(documents)

# Build FAISS vector store
def build_vector_store():
    product_docs = load_product_data("Knowledgebase/Product_metadat.json")
    text_docs = load_text_documents("Knowledgebase/policies/")
    all_docs = product_docs + text_docs
    chunks = split_docs(all_docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("✅ Vector DB created and saved to faiss_index")

if __name__ == "__main__":
    build_vector_store()
