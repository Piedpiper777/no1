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
    last_page_number = None  # 用于记录最后一个有效页码
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            pil_img = page.to_image(resolution=300).original.convert("RGB")
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]

            # OCR 识别
            results = ocr.ocr(np.array(pil_img), cls=False)
            if not results or not results[0]:
                print(f"⚠️ Page {i+1}: 未检测到任何文字")
                continue

            # 改进的页码提取方法
            page_number = extract_page_number_improved(results[0], height, width, i+1, total_pages, last_page_number)
            
            if page_number is None:
                print(f"⚠️ Page {i+1}: 未检测到页码，且无法推断")
                continue
            else:
                # 更新最后一个有效页码
                last_page_number = int(page_number)

            print(f"📄 Page {i+1}: 页码为 {page_number}")

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
                out_path = os.path.abspath(os.path.join(output_dir, f"{label}_page{page_number}.png"))
                cv2.imwrite(out_path, cropped)
                print(f"✅ 提取 {label} (页码: {page_number}) 保存至 {out_path}")


def extract_page_number_improved(ocr_results, height, width, current_page_index, total_pages, last_page_number):
    """
    改进的页码提取方法
    """
    page_candidates = []
    
    # 1. 收集所有可能的页码候选
    for line in ocr_results:
        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
        text = text.strip()
        
        # 获取文本位置信息
        points = line[0]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_center = (min(x_coords) + max(x_coords)) / 2
        y_center = (min(y_coords) + max(y_coords)) / 2
        y_bottom = max(y_coords)
        
        # 多种页码模式匹配
        page_num = None
        confidence = 0
        
        # 模式1: 纯数字
        if re.match(r'^\d+$', text):
            page_num = int(text)
            confidence = 3
        
        # 模式2: 负号+数字 (有时OCR会把页码识别成负号)
        elif re.match(r'^-\d+$', text):
            page_num = int(text[1:])
            confidence = 2
        
        # 模式3: 第X页格式
        elif re.match(r'^第?\s*(\d+)\s*页?$', text):
            match = re.search(r'(\d+)', text)
            if match:
                page_num = int(match.group(1))
                confidence = 4
        
        # 模式4: 包含数字的短文本 (如 "- 5 -", "5.", "Page 5")
        elif len(text) <= 10:
            numbers = re.findall(r'\d+', text)
            if len(numbers) == 1:
                page_num = int(numbers[0])
                confidence = 1
        
        if page_num is not None:
            # 位置评分 (底部中央得分最高)
            position_score = 0
            
            # 垂直位置评分 (底部得分高)
            if y_bottom > height * 0.85:
                position_score += 3
            elif y_bottom > height * 0.75:
                position_score += 2
            elif y_bottom > height * 0.6:
                position_score += 1
            
            # 水平位置评分 (中央得分高)
            if width * 0.4 < x_center < width * 0.6:
                position_score += 2
            elif width * 0.3 < x_center < width * 0.7:
                position_score += 1
            
            # 页码合理性检查
            reasonableness_score = 0
            if 1 <= page_num <= total_pages * 2:  # 允许一定的页码范围
                reasonableness_score = 2
            elif 1 <= page_num <= 1000:  # 基本合理范围
                reasonableness_score = 1
            
            total_score = confidence + position_score + reasonableness_score
            page_candidates.append((page_num, total_score, text))
    
    # 2. 选择最佳候选
    if page_candidates:
        # 按分数排序，选择得分最高的
        page_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = page_candidates[0]
        
        # 如果最高分大于等于4，直接使用
        if best_candidate[1] >= 4:
            return str(best_candidate[0])
        
        # 如果有上一页的页码，检查连续性
        if last_page_number is not None:
            for candidate in page_candidates:
                if abs(candidate[0] - (last_page_number + 1)) <= 1:  # 允许±1的误差
                    return str(candidate[0])
        
        # 否则使用得分最高的候选
        if best_candidate[1] >= 2:
            return str(best_candidate[0])
    
    # 3. 推断方法改进
    if last_page_number is not None:
        # 基于上一页推断
        inferred_page = last_page_number + 1
        
        # 检查推断的页码是否合理
        if inferred_page <= total_pages * 2:  # 合理范围内
            print(f"   推断页码: {inferred_page} (基于上一页: {last_page_number})")
            return str(inferred_page)
    
    # 4. 最后的备用方案：基于PDF页面索引推断
    if current_page_index <= total_pages:
        print(f"   使用PDF页面索引作为页码: {current_page_index}")
        return str(current_page_index)
    
    return None