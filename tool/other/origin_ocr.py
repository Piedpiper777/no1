import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 可以根据实际需求调整语言参数

def extract_text_from_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        print(f"正在处理第{page_num + 1}页")
        page = doc.load_page(page_num)
        
        # 直接尝试从整个页面提取文本
        text = page.get_text("text").strip()
        if not text:
            # 如果提取到的文本为空，则认为是图片型PDF，进行OCR识别
            pix = page.get_pixmap()  # 获取当前页作为图片
            img_path = f"temp_page_{page_num + 1}.png"
            pix.save(img_path)
            ocr_result = ocr.ocr(img_path, cls=True)
            
            # 解析OCR结果
            text = "\n".join([line[1][0] for line in ocr_result[0]])
            print(f"第{page_num + 1}页是图片型PDF")
        else:
            print(f"第{page_num + 1}页是文字型PDF")

        results.append(text)

    # 将结果保存到文件
    with open(output_path, "w", encoding='utf-8') as f:
        for result in results:
            f.write(result + "\n")

# 使用示例
pdf_file = r"/workspace/split_pdfs/CN111964678B/description/description.pdf"  # 输入PDF路径，请替换为你的PDF文件路径
output_file = r"/workspace/tool/chapter_processing/description0.txt"  # 输出文本文件路径，请替换为你想要保存的结果文件路径
extract_text_from_pdf(pdf_file, output_file)