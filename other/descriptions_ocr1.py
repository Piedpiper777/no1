import fitz  # PyMuPDF
import re
import os

INPUT_FILE = r'/workspace/project/split_pdfs/CN111964678B/description/description.pdf'
OUTPUT_FILE = 'output.txt'

# 页眉和页脚裁剪比例
HEADER_CUT_RATIO = 0.07
FOOTER_CUT_RATIO = 0.10

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        top_cut = page_height * HEADER_CUT_RATIO
        bottom_cut = page_height * (1 - FOOTER_CUT_RATIO)

        # 提取去除页眉/页脚区域的文本块
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        filtered_texts = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if y1 < top_cut or y0 > bottom_cut:
                continue  # 在页眉或页脚范围内，跳过
            filtered_texts.append((y0, text.strip()))

        # 按 y 坐标排序（从上到下）
        filtered_texts.sort()
        all_text.extend([text for _, text in filtered_texts])

    return "\n".join(all_text)


def split_paragraphs_by_numbering(text):
    pattern = re.compile(r"(?=\[\d{4}\])")
    segments = pattern.split(text)
    clean_segments = [seg.strip() for seg in segments if seg.strip()]
    return clean_segments


def save_paragraphs(paragraphs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for para in paragraphs:
            f.write(para + '\n\n')


def main():
    print("🟡 提取文本中...")
    raw_text = extract_text_blocks(INPUT_FILE)

    print("🔍 按段落编号拆分...")
    paragraphs = split_paragraphs_by_numbering(raw_text)

    print(f"💾 保存到 {OUTPUT_FILE} ...")
    save_paragraphs(paragraphs, OUTPUT_FILE)

    print(f"✅ 完成！共提取 {len(paragraphs)} 段。")


if __name__ == '__main__':
    main()
