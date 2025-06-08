import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from collections import defaultdict

# 页眉关键词分类
HEADER_KEYWORDS = {
    "drawings": ["附图", "说明书附图", "附", "图", "说明书附"],
    "description": ["说明书", "说", "明"],
    "claims": ["权利要求书", "利", "要", "求"],
    "front": ["国家知识产权局", "国", "家", "知", "识", "产", "局"]
}


def dynamic_extract_header(image, max_height=500, step=100, lang='chi_sim',
                           save_debug_dir=None, save_text_dir=None,
                           doc_name=None, page_idx=None):
    """动态提取页眉，优先匹配更特殊关键词，保存截图+OCR文本"""
    width, height = image.size
    matched_section = "unknown"
    detected_text = ""

    for h in range(step, max_height + step, step):
        crop_box = (0, 0, width, min(h, height))
        region = image.crop(crop_box)
        detected_text = pytesseract.image_to_string(region, lang=lang).strip()

        # 保存页眉截图
        if save_debug_dir and doc_name is not None and page_idx is not None:
            img_out_dir = os.path.join(save_debug_dir, doc_name)
            os.makedirs(img_out_dir, exist_ok=True)
            region.save(os.path.join(img_out_dir, f"page_{page_idx+1}_header_{h}px.png"))

        # 保存OCR文本
        if save_text_dir and doc_name is not None and page_idx is not None:
            txt_out_dir = os.path.join(save_text_dir, doc_name)
            os.makedirs(txt_out_dir, exist_ok=True)
            with open(os.path.join(txt_out_dir, f"page_{page_idx+1}.txt"), "w", encoding="utf-8") as f:
                f.write(detected_text)

        # 优先顺序匹配：drawings > description > claims > front
        for section in ["drawings", "description", "claims", "front"]:
            keywords = HEADER_KEYWORDS.get(section, [])
            for kw in keywords:
                if kw in detected_text:
                    return section, detected_text

    return matched_section, detected_text




def split_pdf_by_dynamic_header(pdf_path, output_root="split_pages",
                                debug_dir="header_debug", text_dir="header_ocr_text",
                                dpi=200):
    """处理单个PDF并按照章节输出图像和OCR"""
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(output_root, doc_name)

    pages = convert_from_path(pdf_path, dpi=dpi)
    section_pages = defaultdict(list)

    for idx, page in enumerate(pages):
        section, header_text = dynamic_extract_header(
            page,
            save_debug_dir=debug_dir,
            save_text_dir=text_dir,
            doc_name=doc_name,
            page_idx=idx
        )
        section_pages[section].append((idx, page))
        print(f"[{doc_name}] 第 {idx+1} 页 → 匹配章节：{section}")

    # 保存每页图像到对应章节目录
    for section, items in section_pages.items():
        section_dir = os.path.join(output_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        for idx, img in items:
            img.save(os.path.join(section_dir, f"page_{idx+1}.png"))

    print(f"[完成] {doc_name} 输出至：{output_dir}")


def batch_process_all_pdfs(input_dir="pdf_files",
                           output_root="split_pages",
                           debug_dir="header_debug",
                           text_dir="header_ocr_text"):
    """批量处理整个文件夹中的PDF"""
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            print(f"\n=== 正在处理：{file} ===")
            split_pdf_by_dynamic_header(
                pdf_path,
                output_root=output_root,
                debug_dir=debug_dir,
                text_dir=text_dir
            )


# === 主入口 ===
if __name__ == "__main__":
    batch_process_all_pdfs(
        input_dir=r"/workspace/pdf_files/test", #pdf输入目录
        output_root="split_pages",     # 图像输出根目录
        debug_dir="header_debug",      # 页眉截图目录
        text_dir="header_ocr_text"     # 页眉OCR文本目录
    )
