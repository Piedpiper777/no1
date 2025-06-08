import os
import re
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)  # 移除 structure 参数

def restructure_paragraphs(text):
    """按专利格式重组段落"""
    pattern = re.compile(r"\[(\d{4})\]")
    matches = list(pattern.finditer(text))

    if not matches:
        return [text.strip()]

    paragraphs = []

    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        para_text = text[start:end].strip()

        if not para_text:
            continue

        lines = [line.strip() for line in para_text.splitlines() if line.strip()]
        if len(lines) == 1:
            paragraphs.append(lines[0])
        elif len(lines) >= 2:
            if lines[-1].endswith("。"):
                paragraphs.append("".join(lines))
            else:
                paragraphs.append("".join(lines[:-1]))
                paragraphs.append(lines[-1])

    return paragraphs

def detect_tables(image_path):
    """
    使用OpenCV检测表格区域（替代PaddleOCR的structure功能）
    返回表格区域的坐标列表
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    thresh = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )
    
    # 检测横线和竖线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # 合并横线和竖线
    table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小区域，保留表格
    table_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 50:  # 过滤小区域
            table_areas.append((x, y, x + w, y + h))
    
    return table_areas

def extract_text_and_tables(pdf_path, output_text_path, table_output_dir):
    """提取PDF中的文本和表格"""
    os.makedirs(table_output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        print(f"正在处理第{page_num + 1}/{len(doc)}页")
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()

        # 生成高分辨率页面图像
        pix = page.get_pixmap(dpi=300)
        img_path = f"temp_page_{page_num + 1}.png"
        pix.save(img_path)

        # 1. 提取文本内容
        ocr_result = ocr.ocr(img_path, cls=True)
        lines = []
        for line in ocr_result[0]:
            if isinstance(line, list) and len(line) >= 2:
                lines.append(line[1][0])
        page_text = "\n".join(lines)
        results.append(page_text)

        # 2. 使用OpenCV检测并提取表格区域
        table_areas = detect_tables(img_path)
        for i, (x1, y1, x2, y2) in enumerate(table_areas):
            img = Image.open(img_path)
            table_img = img.crop((x1, y1, x2, y2))
            table_path = os.path.join(table_output_dir, f"page_{page_num + 1}_table_{i + 1}.jpg")
            table_img.save(table_path)

        print(f"第{page_num + 1}页：提取到表格 {len(table_areas)} 个")

    # 合并所有页面文本并结构化处理
    full_text = "\n".join(results)
    paragraphs = restructure_paragraphs(full_text)

    # 保存结果
    with open(output_text_path, "w", encoding="utf-8") as f:
        for para in paragraphs:
            f.write(para.strip() + "\n\n")

    print(f"\n[🎉] 文本与表格提取完成！文本输出：{output_text_path}，表格图片保存目录：{table_output_dir}")

# 示例用法
if __name__ == "__main__":
    pdf_file = r"/workspace/split_pdfs/CN111964678B/description/description.pdf"
    output_text = r"/workspace/tool/chapter_processing/description.txt"
    table_img_output_dir = r"/workspace/tool/chapter_processing/table_output"
    extract_text_and_tables(pdf_file, output_text, table_img_output_dir)