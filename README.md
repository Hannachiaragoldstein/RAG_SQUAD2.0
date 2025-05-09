# RAG_SQUAD2.0

To create your `README.md` file, you'll want to clearly explain your project — what it does, how it's structured, how to run it, and any setup or dependencies. Based on what you've shared, here's a well-structured `README.md` template you can customize and paste into your repo:

---

````markdown
# Retrieval-Augmented Generation (RAG) System using SQuAD 2.0

This repository implements a modular Retrieval-Augmented Generation (RAG) system using the SQuAD v2.0 dataset. It combines dense passage retrieval with generative question answering using transformer models.

## 📁 Repository Structure

- `retriever.py`: Retrieves relevant context chunks using Sentence Transformers and FAISS.
- `retriever_chunked.py`: Alternative retriever that performs context chunking before embedding.
- `generator.py`: Uses the T5 model to generate answers based on retrieved context.
- `rag_system.py`: Integrates retriever and generator into a full RAG pipeline.
- `squad_evaluate.py`: Official evaluation script for computing EM and F1 on predictions.
- `README.md`: Project overview and instructions.
- `.gitignore`: Files and folders to ignore in version control.

## 🚀 How It Works

1. **Retriever (retriever.py / retriever_chunked.py)**:
   - Loads and optionally chunks the context from the SQuAD dataset.
   - Uses `all-MiniLM-L6-v2` to compute embeddings.
   - Stores embeddings in a FAISS index for efficient similarity search.
   - Returns top-k most relevant passages for a given query.

2. **Generator (generator.py)**:
   - Uses the `t5-small` model to generate an answer given a question and retrieved context.
   - Input is formatted as: `question: <query> context: <retrieved_passage>`

3. **RAG Pipeline (rag_system.py)**:
   - Retrieves context and generates answers.
   - Can be run on individual queries or batches of questions.
   - Stores predictions for evaluation.

4. **Evaluation (squad_evaluate.py)**:
   - Evaluates model predictions using official SQuAD metrics: Exact Match (EM) and F1.

## 🧪 Running the Project

### 1. Install Dependencies

install manually:

```bash
pip install sentence-transformers faiss-cpu transformers tqdm numpy
```

### 2. Prepare the Dataset

Download and place the `dev-v2.0.json` file in your working directory or point to its path in `SQUAD_FILE`.

### 3. Run Retrieval

To generate and save embeddings:

```bash
python retriever.py
```

Or run chunked retrieval:

```bash
python retriever_chunked.py
```

### 4. Run RAG System

```bash
python rag_system.py
```

This will process the dataset and output predictions to a JSON file.

### 5. Evaluate Results

```bash
python squad_evaluate.py path_to_ground_truth.json path_to_predictions.json
```

## 🔧 Configuration

* You can adjust chunk size, overlap, number of retrieved contexts (`top_k`), etc., in the retriever and RAG scripts.
* To sample a portion of the SQuAD dataset for faster experimentation, use the `sample_fraction` parameter.

## 📊 Example

```python
query = "In what country is Normandy located?"
results = search(query, top_k=5)
```

## 📄 License

This project is for academic and research purposes.

```
