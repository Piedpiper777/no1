import os
import glob
import shutil
import sys
from pdf_process_tools.process import run_pdf_processing
from vector_utils import load_texts_from_output, build_faiss_index, model
import pickle
import numpy as np
import json

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
    
    # 统计处理结果
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]  # 去掉.pdf扩展名
        
        # 为每个PDF创建独立的输出目录
        doc_output_dir = os.path.join(output_base_dir, pdf_name)
        
        print(f"\n📄 [{i}/{len(pdf_files)}] 正在处理: {pdf_filename}")
        print(f"   输出目录: {doc_output_dir}")
        
        try:
            # 如果输出目录已存在且包含处理完成的标志，跳过处理
            if is_already_processed(doc_output_dir):
                print(f"   ⏭️ 跳过已处理的文件: {pdf_filename}")
                skipped_count += 1
                continue
            
            # 清理可能存在的不完整输出目录
            if os.path.exists(doc_output_dir):
                shutil.rmtree(doc_output_dir)
            
            # 调用 pdf_processing.py 中的主要处理函数
            print(f"   🔄 开始处理...")
            try:
                success = run_pdf_processing(pdf_path, doc_output_dir)
                print(f"   📋 run_pdf_processing 返回值: {success}")
                print(f"   📁 输出目录是否存在: {os.path.exists(doc_output_dir)}")
                
                if os.path.exists(doc_output_dir):
                    files = os.listdir(doc_output_dir)
                    print(f"   📄 输出目录中的文件: {files}")
                
            except Exception as proc_e:
                print(f"   ❌ run_pdf_processing 抛出异常: {proc_e}")
                success = False
            
            # 验证处理结果
            validation_result = validate_processing_result(doc_output_dir)
            print(f"   🔍 验证结果: {validation_result}")
            
            # 以验证结果为准
            if validation_result:
                print(f"   ✅ 根据验证结果，处理成功: {pdf_filename}")
                success_count += 1
                try:
                    build_vector_index(doc_output_dir)
                except:
                    print(f"   ⚠️ 向量索引建立失败，但不影响主要处理")
            else:
                print(f"   ❌ 根据验证结果，处理失败: {pdf_filename}")
                failed_count += 1
            
        except Exception as e:
            print(f"   ❌ 处理异常: {pdf_filename}, 错误: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            # 清理失败的输出目录
            if os.path.exists(doc_output_dir):
                shutil.rmtree(doc_output_dir, ignore_errors=True)
            continue
    
    # 输出批处理结果统计
    print(f"\n🎉 批量处理完成！")
    print(f"📊 处理统计:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {failed_count}")
    print(f"   ⏭️ 跳过: {skipped_count}")
    print(f"   📁 结果保存在: {output_base_dir}")
    
    # 生成批处理报告
    generate_batch_report(output_base_dir, success_count, failed_count, skipped_count)

def is_already_processed(doc_output_dir):
    """
    检查文档是否已经处理完成
    
    Args:
        doc_output_dir: 文档输出目录
        
    Returns:
        bool: 是否已处理完成
    """
    if not os.path.exists(doc_output_dir):
        return False
    
    # 检查关键文件是否存在
    required_files = [
        "processing_report.json",
        "final_text.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(doc_output_dir, file)):
            return False
    
    # 检查处理报告中的状态
    try:
        report_path = os.path.join(doc_output_dir, "processing_report.json")
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 检查处理状态
        return report.get('processing_status') in ['success', 'partial']
    except:
        return False

def validate_processing_result(doc_output_dir):
    """
    验证处理结果的完整性
    
    Args:
        doc_output_dir: 文档输出目录
        
    Returns:
        bool: 处理结果是否有效
    """
    # 检查基本文件
    if not os.path.exists(os.path.join(doc_output_dir, "processing_report.json")):
        return False
    
    # 检查是否至少有一些有用的输出
    has_text = os.path.exists(os.path.join(doc_output_dir, "final_text.txt"))
    has_images = any(f.endswith('.png') for f in os.listdir(doc_output_dir))
    
    return has_text or has_images

def build_vector_index(doc_output_dir):
    """
    为单个文档建立向量索引
    
    Args:
        doc_output_dir: 文档的输出目录
    """
    print("   🔍 正在建立向量索引...")
    
    try:
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
            "model_path": getattr(model, 'model_name_or_path', "unknown"),
            "created_at": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
        
        with open(os.path.join(index_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 向量索引已保存: {len(texts)} 个文本片段")
        
    except Exception as e:
        print(f"   ⚠️ 向量索引建立失败: {e}")

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

def generate_batch_report(output_base_dir, success_count, failed_count, skipped_count):
    """
    生成批处理报告
    
    Args:
        output_base_dir: 输出基础目录
        success_count: 成功处理的文件数
        failed_count: 失败的文件数
        skipped_count: 跳过的文件数
    """
    report = {
        "batch_processing_summary": {
            "total_files": success_count + failed_count + skipped_count,
            "successful": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "success_rate": f"{success_count/(success_count + failed_count)*100:.1f}%" if (success_count + failed_count) > 0 else "0%"
        },
        "processed_documents": []
    }
    
    # 收集每个处理的文档信息
    for item in os.listdir(output_base_dir):
        item_path = os.path.join(output_base_dir, item)
        if os.path.isdir(item_path):
            doc_report_path = os.path.join(item_path, "processing_report.json")
            if os.path.exists(doc_report_path):
                try:
                    with open(doc_report_path, 'r', encoding='utf-8') as f:
                        doc_report = json.load(f)
                    report["processed_documents"].append({
                        "document_name": item,
                        "status": doc_report.get("processing_status", "unknown"),
                        "sections_found": doc_report.get("sections_found", {}),
                        "files_created": doc_report.get("files_created", {})
                    })
                except:
                    continue
    
    # 保存批处理报告
    batch_report_path = os.path.join(output_base_dir, "batch_processing_report.json")
    with open(batch_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 批处理报告已保存: {batch_report_path}")

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理PDF文件')
    parser.add_argument('input_dir', help='包含PDF文件的输入目录')
    parser.add_argument('-o', '--output', help='输出基础目录（默认为input_dir_output）')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在的输出目录')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在 - {args.input_dir}")
        return 1
    
    # 确定输出目录
    if args.output:
        output_base_dir = args.output
    else:
        dir_name = os.path.basename(os.path.abspath(args.input_dir))
        output_base_dir = f"{dir_name}_output"
    
    # 开始批量处理
    batch_process_pdfs(args.input_dir, output_base_dir)
    return 0

if __name__ == "__main__":
    # 可以直接调用函数或使用命令行
    if len(sys.argv) > 1:
        exit(main())
    else:
        # 示例调用
        INPUT_DIR = "/workspace/no1/test_do"  # 包含PDF文件的目录
        OUTPUT_BASE_DIR = "/workspace/no1/output"  # 输出基础目录
        
        # 开始批量处理
        batch_process_pdfs(INPUT_DIR, OUTPUT_BASE_DIR)
