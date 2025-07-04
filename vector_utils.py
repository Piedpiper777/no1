import faiss
import os
from sentence_transformers import SentenceTransformer

# 推荐的中文编码模型（按优先级排序）
MODEL_PATH = r"/home/zhanggu/Project/tianchi/model/text2vec-base-chinese"  

# 初始化模型
model = SentenceTransformer(MODEL_PATH)


def load_texts_from_output(output_dir):
    """从输出目录加载文本"""
    text_path = os.path.join(output_dir, 'final_text.txt')
    if not os.path.exists(text_path):
        return []
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def build_faiss_index(texts):
    """构建FAISS索引"""
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(texts, question, top_k=3):
    """根据问题检索相关文本"""
    index, _ = build_faiss_index(texts)
    query_vec = model.encode([question])
    distances, indices = index.search(query_vec, top_k)
    results = [texts[i] for i in indices[0]]
    return results
