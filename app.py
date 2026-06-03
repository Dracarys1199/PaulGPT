import streamlit as st
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# Title
st.title("PaulGPT")
st.write("Ask a question and receive an answer based on Paul's epistles.")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# Load ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("paul")

# User question
question = st.text_input("Ask PaulGPT")

if st.button("Ask"):

    if question:

        query_embedding = embed_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        prompt = f"""
You are a scholarly simulation of the Apostle Paul.

You are not literally Paul.

Use the retrieved passages below as your primary source.

Retrieved Passages:
{context}

Question:
{question}

Rules:

- Use only the retrieved passages.
- Do not invent references.
- Do not quote verse numbers.
- Build a logical argument.
- Use Paul's reasoning style.
- Keep under 250 words.
- End with encouragement in Christ.
"""

        response = ollama.chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("PaulGPT")

        st.write(
            response["message"]["content"]
        )

        with st.expander("Retrieved Passages"):

            for i, doc in enumerate(
                results["documents"][0], 1
            ):
                st.markdown(
                    f"### Passage {i}"
                )
                st.write(doc)