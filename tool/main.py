import os
import shutil  # 添加 shutil 导入
import glob    # 添加 glob 导入
from pdf_split import PatentPDFSplitter
from claims_ocr import extract_text_from_pdf
from front import extract_first_page_figure
from draw import extract_figures_by_label
from descriptions_ocr import detect_pdf_type, extract_text_pdf, extract_image_pdf
import fitz  # PyMuPDF

def move_files_to_root(output_dir):
    """将 images 和 tables 文件夹下的文件移动到 output_dir 根目录"""
    folders_to_move = ["images", "tables"]
    
    for folder in folders_to_move:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            continue
        
        # 移动文件夹中的所有文件到根目录
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # 处理命名冲突：添加前缀避免覆盖
                new_filename = f"{folder}_{filename}"
                new_path = os.path.join(output_dir, new_filename)
                
                # 移动文件
                shutil.move(file_path, new_path)
                print(f"已移动: {file_path} -> {new_path}")
        
        # 删除空文件夹
        shutil.rmtree(folder_path)
        print(f"已删除空文件夹: {folder_path}")

def main(pdf_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 分割 PDF 文件
    splitter = PatentPDFSplitter()
    doc = fitz.open(pdf_path)
    page_types = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        img = splitter.pdf_page_to_image(page)
        header_img = splitter.extract_header_region(img)
        texts = splitter.recognize_text(header_img)
        page_type = splitter.classify_page_type_v3(texts)
        page_types.append(page_type)

    # 根据页面类型分组
    grouped_pages = {}
    for i, page_type in enumerate(page_types):
        if page_type not in grouped_pages:
            grouped_pages[page_type] = []
        grouped_pages[page_type].append(i)

    # 保存分割后的 PDF 文件
    split_pdfs = {}
    for page_type, pages in grouped_pages.items():
        if pages:
            output_pdf_path = os.path.join(output_dir, f"{page_type}.pdf")
            new_doc = fitz.open()
            for page_num in pages:
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(output_pdf_path)
            new_doc.close()
            split_pdfs[page_type] = output_pdf_path

    # 处理不同类型的分割文件
    all_text = []
    for page_type, pdf_path in split_pdfs.items():
        if page_type == 'front':
            # 提取第一页附图
            extract_first_page_figure(pdf_path, output_dir)
        elif page_type == 'claims':
            # 提取权利要求书文本
            output_text_path = os.path.join(output_dir, "claims.txt")
            extract_text_from_pdf(pdf_path, output_text_path)
            with open(output_text_path, 'r', encoding='utf-8') as f:
                all_text.extend(f.readlines())
        elif page_type == 'drawings':
            # 提取附图
            extract_figures_by_label(pdf_path, output_dir)
        elif page_type == 'descriptions':
            # 检测 PDF 类型
            pdf_type = detect_pdf_type(pdf_path)
            
            # 创建说明书专用的输出子目录
            desc_output_dir = os.path.join(output_dir, "descriptions")
            os.makedirs(desc_output_dir, exist_ok=True)
            
            if pdf_type == 'text':
                extract_text_pdf(pdf_path, desc_output_dir)
            elif pdf_type == 'image':
                extract_image_pdf(pdf_path, desc_output_dir)
            
            # 合并说明书文本到最终输出
            text_files = glob.glob(os.path.join(desc_output_dir, "text", "*.txt"))
            for txt_file in sorted(text_files):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    all_text.extend(f.readlines())
                all_text.append('\n\n')  # 添加分隔符

    # 合并所有文本到一个文件
    final_text_path = os.path.join(output_dir, "final_text.txt")
    with open(final_text_path, 'w', encoding='utf-8') as f:
        f.writelines(all_text)

    # 移动 images 和 tables 文件夹到根目录
    move_files_to_root(output_dir)
    
    print(f"处理完成，结果保存到: {output_dir}")

if __name__ == "__main__":
    input_pdf_path = r"/workspace/project/pdf_files/CN111964678B.pdf"  # 替换为实际的输入 PDF 路径
    output_dir = "output"
    main(input_pdf_path, output_dir)