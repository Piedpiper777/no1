import os
from PIL import Image
from paddleocr import PaddleOCR

def ocr_claims_with_paddleocr(claims_dir, output_dir, header_height=150, footer_height=200):
    os.makedirs(output_dir, exist_ok=True)
    pages = sorted([f for f in os.listdir(claims_dir) if f.endswith(".png")])

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 初始化 OCR 模型

    full_text = []

    for page in pages:
        page_path = os.path.join(claims_dir, page)
        image = Image.open(page_path)
        width, height = image.size

        # 裁剪正文区域（去除页眉和页脚）
        body_box = (0, header_height, width, height - footer_height)
        body_image = image.crop(body_box)

        # OCR 识别正文区域
        result = ocr.ocr(np.array(body_image), cls=True)
        page_text = "\n".join([line[1][0] for line in result[0]])  # 提取文字部分

        # 保存单页结果
        txt_name = os.path.splitext(page)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page_text)

        full_text.append(page_text)
        print(f"[✓] 完成 OCR：{page}")

    # 合并所有页面结果
    merged_path = os.path.join(output_dir, "claims_text.txt")
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    print(f"\n[🎉] PaddleOCR 完成所有页面，合并结果保存在：{merged_path}")

# 示例调用
if __name__ == "__main__":
    import numpy as np  # 必须引入
    claims_img_dir = r"/workspace/split_pdf/split_pages/CN218108941U/claims"   # 输入路径
    output_text_dir = r"/workspace/output/ocr/claims"                # 输出路径
    ocr_claims_with_paddleocr(claims_img_dir, output_text_dir)
