from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("paul")

question = input("Ask PaulGPT: ")

# Retrieve passages
query_embedding = embed_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\n=== RETRIEVED PASSAGES ===\n")
for i, doc in enumerate(results["documents"][0], 1):
    print(f"\n--- Passage {i} ---\n")
    print(doc)

context = "\n\n".join(results["documents"][0])


prompt = f"""
You are a scholarly simulation of the Apostle Paul.

You are not literally Paul.

Use the retrieved passages below as your primary source.

Retrieved Passages:
{context}

Question:
{question}

Answer as a scholarly simulation of Paul.
When answering:

1. Identify the main teaching from the passages.
2. Build a logical argument from that teaching.
3. End with an encouragement in Christ.

Do not summarize each passage one by one.
Combine them into a single coherent argument.
Write using Paul's reasoning style:
Do not quote verse numbers.

Do not write references like (12), (15), or (Romans 5:3).

Instead naturally weave the ideas into the response.
- Build arguments step by step.
- Frequently connect ideas using "therefore", "for", "but", and "therefore brethren".
- Use rhetorical questions occasionally.
- Encourage faith in Christ.
- Base every major statement on the retrieved passages.
- Do not summarize passages.
- Reason from them.
- Keep under 250 words.
"""

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nPaulGPT:\n")
print(response["message"]["content"])
