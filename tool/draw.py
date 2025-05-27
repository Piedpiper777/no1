import os
import re
import cv2
import pdfplumber
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False, lang='ch')  # 中文 OCR

def extract_figures_by_label(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pil_img = page.to_image(resolution=300).original.convert("RGB")
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]

            # OCR 识别
            results = ocr.ocr(np.array(pil_img), cls=False)
            if not results or not results[0]:
                print(f"⚠️ Page {i+1}: 未检测到任何文字")
                continue

            # 识别图号标签，并获取其顶部和底部位置
            label_boxes = []
            for line in results[0]:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                if re.match(r'^图\s?\d+', text.strip()):
                    try:
                        points = line[0]
                        y_coords = [p[1] for p in points]
                        y_top = min(y_coords)
                        y_bottom = max(y_coords)
                        label_boxes.append((y_top, y_bottom, text.strip().replace(" ", "")))
                    except Exception as e:
                        print(f"⚠️ 坐标解析错误: {e}")
                        continue

            if not label_boxes:
                print(f"⚠️ Page {i+1}: 未检测到图标签")
                continue

            # 按照 y_top 从下往上排序（从页面底部往上）
            label_boxes = sorted(label_boxes, key=lambda x: x[0], reverse=True)

            for idx in range(len(label_boxes)):
                y_top_curr = int(label_boxes[idx][0])
                label = label_boxes[idx][2]

                if idx + 1 < len(label_boxes):
                    y_bottom_prev = int(label_boxes[idx + 1][1])  # 上一个 label 的底部
                else:
                    y_bottom_prev = int(height * 0.07)  # 跳过页眉 

                y_top = max(0, y_bottom_prev)
                y_bottom = min(height, y_top_curr)

                if y_bottom <= y_top:
                    print(f"⚠️ {label}: 无效的裁剪区域 (y_top={y_top}, y_bottom={y_bottom})")
                    continue

                cropped = img_cv[y_top:y_bottom, :]
                out_path = os.path.abspath(os.path.join(output_dir, f"{label}.png"))
                cv2.imwrite(out_path, cropped)
                print(f"✅ 提取 {label} 保存至 {out_path}")

if __name__ == "__main__":
    extract_figures_by_label(
        pdf_path=r"/workspace/project/split_pdfs/CN111964678B/drawings/drawings.pdf",
        output_dir="output_figures"
    )
