import os
import json
import shutil
from pdf_processing import run_pdf_processing
from vector_utils import load_texts_from_output, retrieve

DATA_PATH = "data.json"
DOCUMENTS_DIR = "documents"
OUTPUT_DIR = "output"

def run_pipeline():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for item in dataset:
        pdf_filename = item['document']
        pdf_path = os.path.join(DOCUMENTS_DIR, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"[跳过] 未找到 PDF 文件: {pdf_path}")
            continue

        print(f"\n📄 正在处理: {pdf_filename}")

        # Step 1: 分析 PDF
        run_pdf_processing(pdf_path, OUTPUT_DIR)

        # Step 2: 加载文本
        texts = load_texts_from_output(OUTPUT_DIR)
        if not texts:
            print(f"[警告] 无法提取文本: {pdf_filename}")
            continue

        # Step 3: 检索相关内容
        question = item["question"]
        retrieved = retrieve(texts, question, top_k=3)

        # Step 4: 打印结果
        print("🎯 问题:", question)
        for idx, option in enumerate(item["options"]):
            print(f"{option}")
        print("✅ 正确答案:", item["answer"])
        print("🔍 检索结果:")
        for r in retrieved:
            print("   -", r)

        # Step 5: 清理 output
        #shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    run_pipeline()
