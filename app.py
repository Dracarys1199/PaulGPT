import streamlit as st
import chromadb
from groq import Groq
import os

from sentence_transformers import SentenceTransformer

# ----------------------------
# PAGE
# ----------------------------

st.set_page_config(
    page_title="PaulGPT",
    page_icon="📜",
    layout="centered"
)

st.title("PaulGPT")
st.write(
    "Ask a question and receive an answer based on Paul's epistles."
)

# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ----------------------------
# GROQ CLIENT
# ----------------------------

client_groq = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

# ----------------------------
# CHROMADB
# ----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "paul"
)

# ----------------------------
# USER INPUT
# ----------------------------

question = st.text_input(
    "Ask PaulGPT"
)

# ----------------------------
# ASK BUTTON
# ----------------------------

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching Paul's letters..."):

            # Create query embedding
            query_embedding = embed_model.encode(
                question
            ).tolist()

            # Retrieve passages
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            context = "\n\n".join(
                results["documents"][0]
            )

            # Prompt
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
- Frequently connect ideas using:
  "therefore",
  "for",
  "but",
  "brethren".
- Keep under 250 words.
- End with encouragement in Christ.
"""

            # Groq call
            response = client_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

            st.subheader("PaulGPT")
            st.write(answer)

            # Retrieved passages
            with st.expander("Retrieved Passages"):

                for i, doc in enumerate(
                    results["documents"][0],
                    start=1
                ):
                    st.markdown(
                        f"### Passage {i}"
                    )
                    st.write(doc)

    else:
        st.warning(
            "Please enter a question."
        )
