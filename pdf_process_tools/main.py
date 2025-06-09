import os 
import shutil
import glob
from pdf_process_tools.pdf_split import PatentPDFSplitter
from pdf_process_tools.claims_ocr import extract_text_from_pdf
from pdf_process_tools.front import extract_first_page_figure
from pdf_process_tools.draw import extract_figures_by_label
from pdf_process_tools.descriptions_ocr import detect_pdf_type, extract_text_pdf, extract_image_pdf
import fitz  # PyMuPDF

def flatten_descriptions_output(output_dir):
    """
    将 output/descriptions/ 中的 text.txt 重命名为 descriptions.txt 并移动到 output 根目录，
    并将 images/ 与 tables/ 中的所有文件也移动到根目录并加上 'descriptions_' 前缀。
    最后删除 descriptions 文件夹。
    """
    desc_dir = os.path.join(output_dir, "descriptions")

    # 1. 移动并重命名 text.txt
    text_src = os.path.join(desc_dir, "text.txt")
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

def main(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分割器
    splitter = PatentPDFSplitter(
        use_gpu=False,
        match_algorithm='v3',
        max_chinese_chars=10,
        use_continuity_rules=True  # 启用孤立页纠正机制
    )

    # 分析结构并得到每个章节的页面编号
    sections = splitter.analyze_pdf_structure(pdf_path)

    # 分割 PDF
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
    doc.close()

    # 提取每一部分的信息
    for page_type, pdf_path in split_pdfs.items():
        if page_type == 'front':
            extract_first_page_figure(pdf_path, output_dir)
        elif page_type == 'claims':
            output_text_path = os.path.join(output_dir, "claims.txt")
            extract_text_from_pdf(pdf_path, output_text_path)
        elif page_type == 'drawings':
            extract_figures_by_label(pdf_path, output_dir)
        elif page_type == 'descriptions':
            pdf_type = detect_pdf_type(pdf_path)
            desc_output_dir = os.path.join(output_dir, "descriptions")
            os.makedirs(desc_output_dir, exist_ok=True)
            if pdf_type == 'text':
                extract_text_pdf(pdf_path, desc_output_dir)
            elif pdf_type == 'image':
                extract_image_pdf(pdf_path, desc_output_dir)

    # 移动并整理 descriptions 内容
    flatten_descriptions_output(output_dir)

    # 合并文本：descriptions.txt + claims.txt
    all_text = []
    for fname in ["claims.txt", "descriptions.txt"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                all_text.extend(f.readlines())
            all_text.append('\n\n')

    final_text_path = os.path.join(output_dir, "final_text.txt")
    with open(final_text_path, 'w', encoding='utf-8') as f:
        f.writelines(all_text)

    print(f"处理完成，结果保存到: {output_dir}")

if __name__ == "__main__":
    input_pdf_path = r"/workspace/project/documents/CN212384434U.pdf"  # 替换为你的路径
    output_dir = "output"
    main(input_pdf_path, output_dir)
