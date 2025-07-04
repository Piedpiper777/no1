import os
import glob
import shutil
from pdf_process_tools.main import main as process_pdf
from vector_utils import load_texts_from_output, build_faiss_index, model
import pickle
import numpy as np

def batch_process_pdfs(input_dir, output_base_dir):
    """
    批量处理文件夹内所有PDF文件
    
    Args:
        input_dir: 包含PDF文件的输入目录
        output_base_dir: 输出基础目录，每个PDF会在此目录下创建对应的文件夹
    """
    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 查找所有PDF文件
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ 在 {input_dir} 中未找到PDF文件")
        return
    
    print(f"📁 找到 {len(pdf_files)} 个PDF文件，开始批量处理...")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]  # 去掉.pdf扩展名
        
        # 为每个PDF创建独立的输出目录
        doc_output_dir = os.path.join(output_base_dir, pdf_name)
        
        print(f"\n📄 [{i}/{len(pdf_files)}] 正在处理: {pdf_filename}")
        print(f"   输出目录: {doc_output_dir}")
        
        try:
            # 如果输出目录已存在，跳过处理
            if os.path.exists(doc_output_dir) and os.path.exists(os.path.join(doc_output_dir, 'final_text.txt')):
                print(f"   ⏭️ 跳过已处理的文件: {pdf_filename}")
                continue
            
            # 处理PDF
            process_pdf(pdf_path, doc_output_dir)
            
            # 建立向量索引
            build_vector_index(doc_output_dir)
            
            print(f"   ✅ 完成处理: {pdf_filename}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {pdf_filename}, 错误: {e}")
            # 清理失败的输出目录
            if os.path.exists(doc_output_dir):
                shutil.rmtree(doc_output_dir, ignore_errors=True)
            continue
    
    print(f"\n🎉 批量处理完成！结果保存在: {output_base_dir}")

def build_vector_index(doc_output_dir):
    """
    为单个文档建立向量索引
    
    Args:
        doc_output_dir: 文档的输出目录
    """
    print("   🔍 正在建立向量索引...")
    
    # 加载文本
    texts = load_texts_from_output(doc_output_dir)
    if not texts:
        print("   ⚠️ 未找到文本内容，跳过向量索引建立")
        return
    
    # 建立FAISS索引
    index, embeddings = build_faiss_index(texts)
    
    # 保存索引和相关数据
    index_dir = os.path.join(doc_output_dir, "vector_index")
    os.makedirs(index_dir, exist_ok=True)
    
    # 保存FAISS索引
    import faiss
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    
    # 保存文本和嵌入向量
    with open(os.path.join(index_dir, "texts.pkl"), 'wb') as f:
        pickle.dump(texts, f)
    
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
    
    # 保存元数据
    metadata = {
        "num_texts": len(texts),
        "embedding_dim": embeddings.shape[1],
        "model_path": model.model_name_or_path if hasattr(model, 'model_name_or_path') else "unknown"
    }
    
    with open(os.path.join(index_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 向量索引已保存: {len(texts)} 个文本片段")

def load_vector_index(doc_output_dir):
    """
    加载已保存的向量索引
    
    Args:
        doc_output_dir: 文档的输出目录
    
    Returns:
        tuple: (index, texts, embeddings) 或 None
    """
    index_dir = os.path.join(doc_output_dir, "vector_index")
    
    if not os.path.exists(index_dir):
        return None
    
    try:
        import faiss
        # 加载FAISS索引
        index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        
        # 加载文本
        with open(os.path.join(index_dir, "texts.pkl"), 'rb') as f:
            texts = pickle.load(f)
        
        # 加载嵌入向量
        embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        
        return index, texts, embeddings
    
    except Exception as e:
        print(f"⚠️ 加载向量索引失败: {e}")
        return None

if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "/home/zhanggu/Project/tianchi/test"  # 包含PDF文件的目录
    OUTPUT_BASE_DIR = "/home/zhanggu/Project/tianchi/no1/output"  # 输出基础目录
    
    # 开始批量处理
    batch_process_pdfs(INPUT_DIR, OUTPUT_BASE_DIR)
