import os

# ----------------------------
# 1. Load All Text Files
# ----------------------------
def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                documents.append({
                    "source": filename,
                    "content": text
                })

    return documents


# ----------------------------
# 2. Split Text into Chunks
# ----------------------------
def split_into_chunks(documents, chunk_size=150, overlap=30):
    """
    chunk_size = number of words per chunk
    overlap = overlapping words between chunks (improves RAG accuracy)
    """

    chunks = []

    for doc in documents:
        words = doc["content"].split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "source": doc["source"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            start += chunk_size - overlap
            chunk_id += 1

    return chunks


# ----------------------------
# 3. Run the Process
# ----------------------------
if __name__ == "__main__":

    folder_path = "data/knowledge_base"

    # Load documents
    documents = load_documents(folder_path)
    print(f"Total documents loaded: {len(documents)}")

    # Split documents
    chunks = split_into_chunks(documents, chunk_size=100, overlap=30)
    print(f"Total chunks created: {len(chunks)}")

    # Print sample chunk
    print("\nSample Chunk:")
    print("Source:", chunks[0]["source"])
    print("Chunk ID:", chunks[0]["chunk_id"])
    print("Text Preview:", chunks[0]["text"][:300])


for i in range(3):
    print("\nChunk", i)
    print(chunks[i]["text"])
