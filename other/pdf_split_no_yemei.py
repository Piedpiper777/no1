import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from collections import defaultdict

# 页眉关键词分类
HEADER_KEYWORDS = {
    "drawings": ["附", "图","附图"],
    "description": ["说", "明", "说明书", "说明", "明书"],
    "claims": [ "要", "求", "要求", "权利"],
    "front": [ "国", "家"]
}

# 裁剪参数（像素）
HEADER_HEIGHT_PX = 175
FOOTER_HEIGHT_PX = 200
DPI = 200
POINTS_PER_INCH = 72

def px_to_pt(px, dpi=DPI):
    """将像素值转换为点值(1点=1/72英寸)"""
    return px * POINTS_PER_INCH / dpi

def dynamic_extract_header(image, max_height=500, step=100, lang='chi_sim'):
    """基于 OCR 文本逐字符扫描关键词集，优先返回最先出现的匹配章节"""
    width, height = image.size
    final_text = ""

    for h in range(step, max_height + step, step):
        crop_box = (0, 0, width, min(h, height))
        region = image.crop(crop_box)
        ocr_text = pytesseract.image_to_string(region, lang=lang).strip().replace("\n", "").replace(" ", "")
        final_text = ocr_text

        # 记录每个关键词在文本中的位置
        keyword_hits = []

        for section, keywords in HEADER_KEYWORDS.items():
            for kw in keywords:
                idx = ocr_text.find(kw)
                if idx != -1:
                    keyword_hits.append((idx, kw, section))

        if not keyword_hits:
            continue

        # 按关键词首次出现位置排序
        keyword_hits.sort(key=lambda x: x[0])  # idx 越小越优先

        # 特别处理 description vs drawings 的冲突
        first_hit_idx, first_kw, first_section = keyword_hits[0]
        if first_section == "description":
            # 再查一遍 OCR 文本中是否出现了 drawings 的关键词
            has_drawings = any(kw in ocr_text for kw in HEADER_KEYWORDS["drawings"])
            if has_drawings:
                return "drawings", final_text
            else:
                return "description", final_text

        # 否则直接返回第一个命中的章节
        return first_section, final_text

    return "unknown", final_text



def crop_page_content(page, header_height_px=HEADER_HEIGHT_PX, footer_height_px=FOOTER_HEIGHT_PX):
    """裁剪页面的页眉和页脚，返回处理后的页面矩形"""
    header_height_pt = px_to_pt(header_height_px)
    footer_height_pt = px_to_pt(footer_height_px)
    
    rect = page.rect
    new_rect = fitz.Rect(
        rect.x0,
        rect.y0 + header_height_pt,
        rect.x1,
        rect.y1 - footer_height_pt,
    )
    return new_rect

def split_pdf_by_dynamic_header(pdf_path, output_root="split_pdfs"):
    """处理单个PDF并按照章节输出为独立PDF文件，同时裁剪非首页的页眉页脚"""
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(output_root, doc_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{doc_name}_ocr_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    section_ranges = defaultdict(list)
    current_section = None
    current_section_start = 0

    for idx in range(total_pages):
        page = pdf_document.load_page(idx)
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        section, ocr_text = dynamic_extract_header(img)

        # 日志记录
        log_entry = f"第 {idx + 1} 页 - 匹配章节: {section}\nOCR结果:\n{ocr_text}\n{'-' * 40}\n"
        log_file.write(log_entry)

        print(f"[{doc_name}] 第 {idx+1}/{total_pages} 页 → 匹配章节：{section}")
        
        if current_section is not None and section != current_section:
            section_ranges[current_section].append((current_section_start, idx - 1))
            current_section_start = idx
        
        current_section = section

    if current_section is not None:
        section_ranges[current_section].append((current_section_start, total_pages - 1))

    log_file.close()

    for section, ranges in section_ranges.items():
        section_dir = os.path.join(output_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        
        for i, (start, end) in enumerate(ranges):
            if start > end:
                continue

            new_pdf = fitz.open()
            
            for page_num in range(start, end + 1):
                new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                
                if page_num != start:  # 当前章节第一页不裁剪
                    new_page = new_pdf[-1]
                    new_rect = crop_page_content(new_page)
                    new_page.set_cropbox(new_rect)

            output_filename = f"pages_{start+1}-{end+1}.pdf"
            if len(ranges) == 1:
                output_filename = f"{section}.pdf"
            elif i > 0:
                output_filename = f"{section}_{i+1}.pdf"

            output_path = os.path.join(section_dir, output_filename)
            new_pdf.save(output_path)
            new_pdf.close()
            print(f"  已保存: {output_path}")
    
    pdf_document.close()
    print(f"[完成] {doc_name} 输出至：{output_dir}")

def batch_process_all_pdfs(input_dir="pdf_files", output_root="split_pdfs"):
    """批量处理整个文件夹中的PDF"""
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            print(f"\n=== 正在处理：{file} ===")
            split_pdf_by_dynamic_header(pdf_path, output_root)

# === 主入口 ===
if __name__ == "__main__":
    batch_process_all_pdfs(
        input_dir=r"/workspace/pdf_files/test",
        output_root=r"/workspace/split_pdfs"
    )
