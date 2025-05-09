import os
import json
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# =============================
# Set Random Seed for Reproducibility
# =============================
SEED = 42
np.random.seed(SEED)

# =============================
# Chunking Function
# =============================
def chunk_text(text: str, chunk_size: int = 100, overlap: int = 35) -> List[str]:
    """Split text into overlapping chunks of fixed size."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =============================
# Load SQuAD Dataset
# =============================
def load_squad_dataset(filepath: str):
    """Load all contexts, questions, and answers from the SQuAD dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SQuAD dataset file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    contexts, questions, answers = [], [], []

    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            chunks = chunk_text(context)
            contexts.extend(chunks)
            for qa in paragraph["qas"]:
                questions.append(qa["question"])
                answers.append({
                    "text": [answer["text"] for answer in qa.get("answers", [])],
                    "answer_start": [answer["answer_start"] for answer in qa.get("answers", [])],
                    "is_impossible": qa.get("is_impossible", False)
                })

    return contexts, questions, answers

# =============================
# Load Data & Compute Embeddings
# =============================
DATA_DIR = ""  # Set this to your dataset directory
SQUAD_FILE = os.path.join(DATA_DIR, "sampled_dev-v2.0.json")
EMBEDDING_FILE = "embeddings_sentencetransformers_chunked100_overlap35.npy"

contexts, questions, answers = load_squad_dataset(SQUAD_FILE)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts: List[str], batch_size=16):
    """Compute embeddings for a list of texts using the transformer model."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True, batch_size=batch_size)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

if os.path.exists(EMBEDDING_FILE):
    embeddings = np.load(EMBEDDING_FILE, allow_pickle=True)
else:
    embeddings = get_embeddings(contexts)
    np.save(EMBEDDING_FILE, embeddings)

# =============================
# Index the Embeddings using FAISS
# =============================
index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# =============================
# Search Function
# =============================
def search(query: str, top_k=5) -> List[Tuple[str, float]]:
    """Find the most relevant chunks with unique similarity scores."""
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, len(contexts))

    unique_scores = set()
    results = []

    for i, idx in enumerate(indices[0]):
        chunk = contexts[idx]
        score = round(distances[0][i], 4)

        if score not in unique_scores:
            unique_scores.add(score)
            results.append((chunk, score))

        if len(results) == top_k:
            break

    return results
