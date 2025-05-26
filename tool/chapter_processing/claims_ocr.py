import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import os

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def is_text_based(page):
    return bool(page.get_text("text").strip())

def crop_ocr_area(pil_img, header_ratio=0.07, footer_ratio=0.10):
    width, height = pil_img.size
    top = int(height * header_ratio)
    bottom = int(height * (1 - footer_ratio))
    return pil_img.crop((0, top, width, bottom))

def extract_lines_with_indent(page, header_ratio=0.07, footer_ratio=0.10, indent_threshold=10):
    blocks = page.get_text("dict")["blocks"]
    page_rect = page.rect
    top = page_rect.y0 + page_rect.height * header_ratio
    bottom = page_rect.y1 - page_rect.height * footer_ratio

    lines = []
    for block in blocks:
        if "lines" not in block:
            continue  # 跳过非文本块

        for line in block["lines"]:
            if not line["spans"]:
                continue

            first_span = line["spans"][0]
            x0, y0 = first_span["bbox"][0], first_span["bbox"][1]

            if not (top <= y0 <= bottom):
                continue

            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue

            is_indented = x0 > indent_threshold
            lines.append((text, is_indented))

    return lines


def smart_join_lines_with_indent(lines):
    paragraphs = []
    paragraph = ""

    for idx, (line, is_indented) in enumerate(lines):
        if not line:
            continue

        if is_indented:
            if paragraph:
                paragraphs.append(paragraph.strip())
            paragraph = line
        else:
            if paragraph and not paragraph.endswith(("。", "！", "？", "；")):
                paragraph += line
            else:
                if paragraph:
                    paragraphs.append(paragraph.strip())
                paragraph = line

    if paragraph:
        paragraphs.append(paragraph.strip())

    return paragraphs

def ocr_paragraph_rebuild(ocr_result, line_gap_threshold=15):
    if not ocr_result or not ocr_result[0]:
        return []

    lines = [(line[1][0], line[0][1][1]) for line in ocr_result[0]]
    lines.sort(key=lambda x: x[1])

    paragraphs = []
    paragraph = ""
    last_y = None

    for text, y in lines:
        if last_y is None or abs(y - last_y) < line_gap_threshold:
            paragraph += text.strip() + " "
        else:
            paragraphs.append(paragraph.strip())
            paragraph = text.strip()
        last_y = y

    if paragraph:
        paragraphs.append(paragraph.strip())

    return paragraphs

def merge_cross_page_paragraphs(all_paragraphs):
    """跨页段落合并逻辑"""
    merged = []
    buffer = ""

    for i in range(len(all_paragraphs)):
        para = all_paragraphs[i].strip()
        if not para: # 跳过空字符串，这很重要，特别是您移除了 raw_paragraphs.append("") 后
            continue

        # 如果上一段不以句号等结尾，当前段不缩进，拼接
        # 并且确保 merged 列表不为空
        if merged:
            last = merged[-1]
            # 这里的逻辑是为了确保连续的文本能够合并，如果上一段不以句号等结尾
            # 并且当前段落不是显式缩进（起始不带空格），则合并
            if not last.endswith(("。", "！", "？", "；")) and not para.startswith((" ", "\u3000", "　")):
                merged[-1] = last + para  # 拼接
                continue

        merged.append(para)

    return merged

def extract_text_from_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    raw_paragraphs = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        print(f"正在处理第{page_num + 1}页...")

        if is_text_based(page):
            print("文字型PDF")
            lines = extract_lines_with_indent(page)
            paragraphs = smart_join_lines_with_indent(lines)
        else:
            print("图片型PDF")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            cropped_img = crop_ocr_area(img)
            ocr_result = ocr.ocr(np.array(cropped_img), cls=True)
            paragraphs = ocr_paragraph_rebuild(ocr_result)

        raw_paragraphs.extend(paragraphs)
        # raw_paragraphs.append("")  # 移除或注释掉此行，它会导致额外的空行

    # 合并跨页自然段
    # 在调用 merge_cross_page_paragraphs 之前，最好清理一下 raw_paragraphs 中的空字符串，
    # 以免影响后续的合并逻辑，尤其是在移除了 append("") 之后。
    # 虽然 merge_cross_page_paragraphs 中有 `if not para: continue`，
    # 但提前清理可以使数据更纯净。
    # filtered_paragraphs = [p for p in raw_paragraphs if p.strip()] # 考虑是否需要这一步
    # final_paragraphs = merge_cross_page_paragraphs(filtered_paragraphs)
    final_paragraphs = merge_cross_page_paragraphs(raw_paragraphs)


    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_paragraphs)) # 将 \n\n 修改为 \n

    print(f"\n提取完成，保存到: {output_path}")

if __name__ == "__main__":
    input_pdf_path = r"/workspace/project/split_pdfs/CN111964678B/claims/claims.pdf"
    output_text_path = "claims.txt"

    if not os.path.exists(input_pdf_path):
        print(f"错误：文件 {input_pdf_path} 不存在")
    else:
        extract_text_from_pdf(input_pdf_path, output_text_path)