from sentence_transformers import SentenceTransformer
import faiss
import os

MODEL_PATH = r"/workspace/project/model/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_PATH)

def load_texts_from_output(output_dir):
    text_path = os.path.join(output_dir, 'descriptions.txt')
    if not os.path.exists(text_path):
        return []
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def build_faiss_index(texts):
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(texts, question, top_k=3):
    index, _ = build_faiss_index(texts)
    query_vec = model.encode([question])
    distances, indices = index.search(query_vec, top_k)
    results = [texts[i] for i in indices[0]]
    return results
