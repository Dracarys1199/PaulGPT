from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data/paul_letters.txt","r",encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

import re

sections = re.split(r"(?=Book:)", text)

chunks = []

for section in sections:
    if section.strip():
        section_chunks = splitter.split_text(section)
        chunks.extend(section_chunks)

print(len(chunks))


from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

try:
    client.delete_collection("paul")
except:
    pass

collection = client.get_or_create_collection(
    name="paul"
)


for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk
    ).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )