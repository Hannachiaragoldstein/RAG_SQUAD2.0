import json
from tqdm import tqdm  
from retriever_chunked import search
from generator_qa import generate_answer

# =============================
# Function: RAG System
# =============================
def rag_system(query, top_k=5):
    """Retrieve top-k contexts and generate an answer."""
    return generate_answer(query, search(query, top_k))

# =============================
# Load & Process SQuAD Dataset
# =============================
SQUAD_FILE = "sampled_dev-v2.0.json"  # Change this if using full dev-v2.0.json
PREDICTIONS_FILE = "predictions_sentancetransformers_chunked100_overlap35_qa.json"

# Load full dataset
with open(SQUAD_FILE, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# Extract all questions from the dataset
questions = []
for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            questions.append({"id": qa["id"], "question": qa["question"]})

print(f"Loaded {len(questions)} questions from the dataset.")

# =============================
# Run RAG on All Questions
# =============================
predictions = {
    q['id']: rag_system(q['question'])
    for q in tqdm(questions, desc="Processing questions", unit="question")
}

# =============================
# Save Predictions
# =============================
with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print(f"Predictions saved to {PREDICTIONS_FILE}")
