import os
import json
import re
import glob
from vector_utils import load_texts_from_output, retrieve
from llm_utils import call_llm_with_context

DATA_PATH = "data.json"
OUTPUT_DIR = "output"

def extract_page_numbers(question):
    """从问题中提取页码信息"""
    page_patterns = [
        r'第(\d+)页',
        r'第(\d+)页的',
        r'页码(\d+)',
        r'(\d+)页'
    ]
    
    pages = []
    for pattern in page_patterns:
        matches = re.findall(pattern, question)
        pages.extend([int(match) for match in matches])
    
    return list(set(pages))  # 去重

def find_figure_files(doc_output_dir, pages):
    """根据页码查找对应的图表文件"""
    figure_files = []
    
    if not pages:
        return figure_files
    
    for page_num in pages:
        # 查找包含指定页码的图表文件
        pattern = os.path.join(doc_output_dir, f"*page{page_num}*.png")
        files = glob.glob(pattern)
        figure_files.extend(files)
    
    return figure_files

def process_single_item(item):
    """处理单个数据项"""
    pdf_filename = item['document']
    
    # 获取对应文档的输出目录（已预处理完成）
    doc_output_dir = os.path.join(OUTPUT_DIR, pdf_filename.replace('.pdf', ''))
    
    if not os.path.exists(doc_output_dir):
        print(f"[错误] 未找到预处理结果: {doc_output_dir}")
        return None

    print(f"\n📄 正在处理: {pdf_filename}")

    # Step 1: 加载已构建的向量数据库和文本
    texts = load_texts_from_output(doc_output_dir)
    if not texts:
        print(f"[警告] 无法加载文本数据: {pdf_filename}")
        return None

    # Step 2: 基于问题检索相关文本
    question = item["question"]
    retrieved_texts = retrieve(texts, question, top_k=5)

    # Step 3: 检查问题中是否包含页码信息，如有则检索对应图片
    pages = extract_page_numbers(question)
    figure_files = []
    
    if pages:
        print(f"🔍 检测到页码: {pages}")
        figure_files = find_figure_files(doc_output_dir, pages)
        if figure_files:
            print(f"📊 找到图表文件: {[os.path.basename(f) for f in figure_files]}")
        else:
            print(f"⚠️ 未找到页码 {pages} 对应的图表文件")

    # Step 4: 调用大模型
    response = call_llm_with_context(
        question=question,
        options=item["options"],
        retrieved_texts=retrieved_texts,
        figure_files=figure_files
    )
    
    # Step 5: 打印结果
    print_results(item, retrieved_texts, figure_files, response)
    
    return response

def print_results(item, retrieved_texts, figure_files, response):
    """打印处理结果"""
    print("🎯 问题:", item["question"])
    for option in item["options"]:
        print(f"   {option}")
    print("✅ 正确答案:", item["answer"])
    
    print("🔍 检索到的文本:")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"   {i}. {text[:100]}..." if len(text) > 100 else f"   {i}. {text}")
    
    if figure_files:
        print("📊 相关图表:")
        for fig in figure_files:
            print(f"   - {os.path.basename(fig)}")
    
    print("🤖 模型回答:", response)
    print("-" * 80)

def run_pipeline():
    """运行完整流水线"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        result = process_single_item(item)
        if result:
            results.append({
                'id': item.get('id'),
                'question': item['question'],
                'options': item['options'],
                'model_answer': result, 
                'document': item['document']
            })
    
    # 保存结果
    with open('pipeline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 处理完成，结果已保存到 pipeline_results.json")

if __name__ == "__main__":
    run_pipeline()
