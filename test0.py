import fitz  # PyMuPDF
import re
import os

INPUT_FILE = r'/workspace/project/intput/description.pdf'
OUTPUT_FILE = 'output/output.txt'

# 页眉和页脚裁剪比例
HEADER_CUT_RATIO = 0.07
FOOTER_CUT_RATIO = 0.10

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        
        # 首页不过滤页眉（保留大标题和小标题）
        if page_num == 0:
            top_cut = 0
            bottom_cut = page_height * (1 - FOOTER_CUT_RATIO)
        else:
            # 其他页面按原规则过滤
            top_cut = page_height * HEADER_CUT_RATIO
            bottom_cut = page_height * (1 - FOOTER_CUT_RATIO)
        
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            # 仅过滤页脚（首页不过滤页眉）
            if y0 > bottom_cut:
                continue
            all_text.append((y0, text.strip()))
        
        # 按 y 坐标排序
        all_text.sort(key=lambda x: x[0])
    
    return "\n".join([text for _, text in all_text])

def extract_paragraphs(text):
    # 匹配 [数字] 格式的段落
    pattern = re.compile(r"(\[\d{4}\].*?)(?=\[\d{4}\]|$)", re.DOTALL)
    
    # 提取所有带 [数字] 标记的段落
    matches = pattern.finditer(text)
    paragraphs = [match.group(1).strip() for match in matches]
    
    # 提取大标题和小标题（假设它们在 [0001] 之前）
    first_para_pos = text.find("[0001]")
    if first_para_pos > 0:
        headers = text[:first_para_pos].strip()
        if headers:
            # 将标题按空行分割成多个段落
            headers_paras = [p.strip() for p in headers.split("\n\n") if p.strip()]
            paragraphs = headers_paras + paragraphs
    
    return paragraphs

def save_paragraphs(paragraphs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for para in paragraphs:
            f.write(para + '\n\n')

def main():
    print("🟡 提取文本中...")
    raw_text = extract_text_blocks(INPUT_FILE)
    
    print("🔍 按段落编号拆分...")
    paragraphs = extract_paragraphs(raw_text)
    
    print(f"💾 保存到 {OUTPUT_FILE} ...")
    save_paragraphs(paragraphs, OUTPUT_FILE)
    
    print(f"✅ 完成！共提取 {len(paragraphs)} 段。")

if __name__ == '__main__':
    main()