import os
import sys
import shutil
import fitz  # PyMuPDF

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_split import PatentPDFSplitter
from claims_ocr import extract_text_from_pdf as extract_claims_text
from front import extract_first_page_figure
from draw import extract_figures_by_label
from descriptions_ocr import detect_pdf_type, extract_text_pdf, extract_image_pdf
import json

def flatten_descriptions_output(output_dir):
    """
    将 output/descriptions/ 中的 text.txt 重命名为 descriptions.txt 并移动到 output 根目录，
    并将 images/ 与 tables/ 中的所有文件也移动到根目录并加上 'descriptions_' 前缀。
    最后删除 descriptions 文件夹。
    """
    desc_dir = os.path.join(output_dir, "descriptions")
    
    if not os.path.exists(desc_dir):
        return

    # 1. 移动并重命名 text.txt
    text_src = os.path.join(desc_dir, "descriptions.txt")
    text_dst = os.path.join(output_dir, "descriptions.txt")
    if os.path.exists(text_src):
        shutil.move(text_src, text_dst)
        print(f"已移动并重命名: {text_src} -> {text_dst}")

    # 2. 移动 images/ 和 tables/ 下所有文件，加前缀
    for subfolder in ["images", "tables"]:
        folder_path = os.path.join(desc_dir, subfolder)
        if not os.path.exists(folder_path):
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, folder_path).replace(os.sep, "_")
                dst_file = os.path.join(output_dir, f"descriptions_{rel_path}")
                shutil.move(src_file, dst_file)
                print(f"已移动: {src_file} -> {dst_file}")

    # 3. 删除整个 descriptions 文件夹
    if os.path.exists(desc_dir):
        shutil.rmtree(desc_dir)
        print(f"已删除目录: {desc_dir}")

def run_pdf_processing(pdf_path, output_dir):
    """
    运行PDF文件的全流程处理
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
    """
    print(f"🚀 开始处理PDF文件: {os.path.basename(pdf_path)}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 初始化分割器并分析PDF结构
    print("\n📊 Step 1: 分析PDF结构...")
    splitter = PatentPDFSplitter(
        use_gpu=False,
        match_algorithm='v3',
        max_chinese_chars=10,
        use_continuity_rules=True
    )

    # 分析结构并得到每个章节的页面编号
    sections = splitter.analyze_pdf_structure(pdf_path)
    
    # 检查是否有页面被分类
    total_pages = sum(len(pages) for pages in sections.values())
    if total_pages == 0:
        print("❌ 未能识别任何页面类型，处理失败")
        return False

    # Step 2: 分割PDF
    print("\n✂️ Step 2: 分割PDF...")
    doc = fitz.open(pdf_path)
    split_pdfs = {}
    
    for section_type, pages in sections.items():
        if pages:
            output_pdf_path = os.path.join(output_dir, f"{section_type}.pdf")
            new_doc = fitz.open()
            for page_num in sorted(pages):
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(output_pdf_path)
            new_doc.close()
            split_pdfs[section_type] = output_pdf_path
            print(f"  ✅ 生成 {section_type}.pdf (页面: {[p+1 for p in pages]})")
    doc.close()

    # Step 3: 提取各部分信息
    print("\n🔍 Step 3: 提取各部分信息...")
    
    text_files_created = []
    
    for page_type, pdf_path in split_pdfs.items():
        print(f"\n  处理 {page_type} 部分...")
        
        try:
            if page_type == 'front':
                print("    - 提取首页图像...")
                extract_first_page_figure(pdf_path, output_dir)
                
            elif page_type == 'claims':
                print("    - 提取权利要求...")
                output_text_path = os.path.join(output_dir, "claims.txt")
                extract_claims_text(pdf_path, output_text_path)
                if os.path.exists(output_text_path):
                    text_files_created.append("claims.txt")
                    
            elif page_type == 'drawings':
                print("    - 提取附图...")
                extract_figures_by_label(pdf_path, output_dir)
                
            elif page_type == 'descriptions':
                print("    - 提取说明书...")
                pdf_type = detect_pdf_type(pdf_path)
                print(f"      检测到PDF类型: {pdf_type}")
                
                desc_output_dir = os.path.join(output_dir, "descriptions")
                os.makedirs(desc_output_dir, exist_ok=True)
                
                if pdf_type == 'text':
                    extract_text_pdf(pdf_path, desc_output_dir)
                elif pdf_type == 'image':
                    extract_image_pdf(pdf_path, desc_output_dir)
                else:  # mixed
                    print("      混合型PDF，使用文本模式处理")
                    extract_text_pdf(pdf_path, desc_output_dir)
                    
        except Exception as e:
            print(f"    ❌ 处理 {page_type} 时出错: {e}")
            continue

    # Step 4: 整理descriptions内容
    print("\n📋 Step 4: 整理文档结构...")
    flatten_descriptions_output(output_dir)
    
    # 检查descriptions.txt是否创建成功
    desc_file = os.path.join(output_dir, "descriptions.txt")
    if os.path.exists(desc_file):
        text_files_created.append("descriptions.txt")

    # Step 5: 合并所有文本
    print("\n🔗 Step 5: 合并文本内容...")
    all_text = []
    
    for fname in ["claims.txt", "descriptions.txt"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_text.append(f"=== {fname.replace('.txt', '').upper()} ===\n")
                    all_text.append(content)
                    all_text.append('\n\n')
                    print(f"  ✅ 合并 {fname}")
        else:
            print(f"  ⚠️ 未找到 {fname}")

    # 保存合并后的文本
    if all_text:
        final_text_path = os.path.join(output_dir, "final_text.txt")
        with open(final_text_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_text))
        print(f"  ✅ 创建 final_text.txt")
    else:
        print("  ⚠️ 没有找到任何文本内容")

    # Step 6: 生成处理报告
    print("\n📊 Step 6: 生成处理报告...")
    
    # 统计文件
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    text_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    report = {
        "input_file": os.path.basename(pdf_path),
        "output_directory": output_dir,
        "sections_found": {k: len(v) for k, v in sections.items() if v},
        "files_created": {
            "pdf_files": len(pdf_files),
            "image_files": len(image_files), 
            "text_files": len(text_files)
        },
        "text_files_list": text_files,
        "processing_status": "success" if all_text else "partial"
    }
    
    # 保存报告
    report_path = os.path.join(output_dir, "processing_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 处理完成！")
    print(f"📄 PDF部分: {', '.join(k for k, v in sections.items() if v)}")
    print(f"📁 生成文件: {len(pdf_files)} PDF, {len(image_files)} 图片, {len(text_files)} 文本")
    print(f"📊 处理报告: {report_path}")
    

def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF文件全流程自动处理工具')
    parser.add_argument('pdf_path', help='输入PDF文件路径')
    parser.add_argument('-o', '--output', help='输出目录路径（默认为PDF文件名_output）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出目录')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.pdf_path):
        print(f"❌ 错误: 输入文件不存在 - {args.pdf_path}")
        return 1
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = f"{pdf_name}_output"
    
    # 检查输出目录
    if os.path.exists(output_dir):
        if not args.overwrite:
            response = input(f"输出目录已存在: {output_dir}\n是否覆盖? (y/N): ")
            if response.lower() != 'y':
                print("取消处理")
                return 0
        shutil.rmtree(output_dir)
    
    # 执行处理
    try:
        success = run_pdf_processing(args.pdf_path, output_dir)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1

if __name__ == "__main__":
    # 可以直接调用函数或使用命令行
    if len(sys.argv) > 1:
        exit(main())
    else:
        # 示例调用
        pdf_path = "/workspace/no1/test_do/CN212149980U.pdf"
        output_dir = "output_test"
        run_pdf_processing(pdf_path, output_dir)
    

def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF文件全流程自动处理工具')
    parser.add_argument('pdf_path', help='输入PDF文件路径')
    parser.add_argument('-o', '--output', help='输出目录路径（默认为PDF文件名_output）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出目录')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.pdf_path):
        print(f"❌ 错误: 输入文件不存在 - {args.pdf_path}")
        return 1
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = f"{pdf_name}_output"
    
    # 检查输出目录
    if os.path.exists(output_dir):
        if not args.overwrite:
            response = input(f"输出目录已存在: {output_dir}\n是否覆盖? (y/N): ")
            if response.lower() != 'y':
                print("取消处理")
                return 0
        shutil.rmtree(output_dir)
    
    # 执行处理
    try:
        success = run_pdf_processing(args.pdf_path, output_dir)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1

