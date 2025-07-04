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

            # OCR 识别 - 分两部分进行
            # 1. 全页面OCR，用于识别图标签
            full_results = ocr.ocr(np.array(pil_img), cls=False)
            if not full_results or not full_results[0]:
                print(f"⚠️ Page {i+1}: 未检测到任何文字")
                continue

            # 2. 页面底部30%区域OCR，专门用于页码识别
            footer_start = int(height * 0.7)  # 从70%位置开始到底部
            footer_img = np.array(pil_img)[footer_start:, :]  # 裁剪底部30%区域
            
            print(f"📄 Page {i+1}: 页面尺寸 {width}x{height}, 页码识别区域: {footer_start}-{height}")
            
            footer_results = ocr.ocr(footer_img, cls=False)
            
            # 调整页脚OCR结果的坐标，因为我们裁剪了图像
            adjusted_footer_results = []
            if footer_results and footer_results[0]:
                for line in footer_results[0]:
                    # 调整坐标，加上footer_start偏移量
                    adjusted_points = [[x, y + footer_start] for x, y in line[0]]
                    adjusted_line = [adjusted_points, line[1]]
                    adjusted_footer_results.append(adjusted_line)

            # 页码提取 - 只使用底部30%区域的OCR结果
            page_number = extract_page_number_from_footer(adjusted_footer_results, height, width, i+1, total_pages, last_page_number)
            
            if page_number is None:
                print(f"⚠️ Page {i+1}: 未检测到页码，且无法推断")
                continue
            else:
                # 更新最后一个有效页码
                last_page_number = int(page_number)

            print(f"📄 Page {i+1}: 页码为 {page_number}")

            # 识别图号标签 - 使用全页面OCR结果
            label_boxes = []
            for line in full_results[0]:
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
                    y_bottom_prev = int(height * 0.2)  # 跳过页眉 

                y_top = max(0, y_bottom_prev)
                y_bottom = min(height, y_top_curr)

                if y_bottom <= y_top:
                    print(f"⚠️ {label}: 无效的裁剪区域 (y_top={y_top}, y_bottom={y_bottom})")
                    continue

                cropped = img_cv[y_top:y_bottom, :]
                out_path = os.path.abspath(os.path.join(output_dir, f"{label}_page{page_number}.png"))
                cv2.imwrite(out_path, cropped)
                print(f"✅ 提取 {label} (页码: {page_number}) 保存至 {out_path}")


def extract_page_number_from_footer(footer_ocr_results, height, width, current_page_index, total_pages, last_page_number):
    """
    从页脚OCR结果中提取页码（专门用于底部30%区域）
    """
    if not footer_ocr_results:
        print(f"   页脚区域未检测到文字")
        return fallback_page_number(current_page_index, total_pages, last_page_number)
    
    page_candidates = []
    footer_start = int(height * 0.7)  # 页脚开始位置
    
    print(f"   🔍 页脚区域OCR结果 (y范围: {footer_start}-{height}):")
    
    # 1. 收集页脚区域的所有可能页码候选
    for line in footer_ocr_results:
        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
        text = text.strip()
        
        # 获取文本位置信息
        points = line[0]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_center = (min(x_coords) + max(x_coords)) / 2
        y_center = (min(y_coords) + max(y_coords)) / 2
        y_bottom = max(y_coords)
        
        print(f"     文本: '{text}' @ ({x_center:.0f}, {y_center:.0f})")
        
        # 多种页码模式匹配
        page_num = None
        confidence = 0
        
        # 模式1: 纯数字 (最常见)
        if re.match(r'^\d{1,3}$', text):
            page_num = int(text)
            confidence = 5  # 页脚区域的纯数字，给更高置信度
        
        # 模式2: 负号+数字 (有时OCR会把页码识别成负号)
        elif re.match(r'^-\d{1,3}$', text):
            page_num = int(text[1:])
            confidence = 4
        
        # 模式3: 第X页格式
        elif re.match(r'^第?\s*(\d{1,3})\s*页?$', text):
            match = re.search(r'(\d+)', text)
            if match:
                page_num = int(match.group(1))
                confidence = 6  # 明确的页码格式，最高置信度
        
        # 模式4: 带横线格式 (如 "- 5 -", "5.", "Page 5")
        elif len(text) <= 10:
            numbers = re.findall(r'\d+', text)
            if len(numbers) == 1:
                num = int(numbers[0])
                if 1 <= num <= 999:  # 合理页码范围
                    page_num = num
                    confidence = 3
        
        if page_num is not None and 1 <= page_num <= 999:
            # 页脚区域位置评分 (在页脚区域内，位置都比较理想)
            position_score = 0
            
            # 垂直位置评分 (越靠近底部得分越高)
            y_ratio = y_center / height
            if y_ratio > 0.9:  # 最底部10%
                position_score += 3
            elif y_ratio > 0.8:  # 底部20%
                position_score += 2
            else:  # 底部30%其他位置
                position_score += 1
            
            # 水平位置评分 (页码通常在中央或右侧)
            x_ratio = x_center / width
            if 0.4 <= x_ratio <= 0.6:  # 中央
                position_score += 2
            elif x_ratio > 0.7:  # 右侧
                position_score += 1
            
            # 页码合理性检查
            reasonableness_score = 0
            if 1 <= page_num <= total_pages * 2:  # 允许一定的页码范围
                reasonableness_score = 2
            elif 1 <= page_num <= 1000:  # 基本合理范围
                reasonableness_score = 1
            
            total_score = confidence + position_score + reasonableness_score
            page_candidates.append((page_num, total_score, text))
            print(f"     ✓ 页码候选: '{text}' -> {page_num} (置信度:{confidence}, 位置:{position_score}, 合理性:{reasonableness_score}, 总分:{total_score})")
    
    # 2. 选择最佳候选
    if page_candidates:
        # 按分数排序，选择得分最高的
        page_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = page_candidates[0]
        
        print(f"   🎯 最佳候选: {best_candidate[0]} (总分: {best_candidate[1]})")
        
        # 如果最高分大于等于6，直接使用
        if best_candidate[1] >= 6:
            return str(best_candidate[0])
        
        # 如果有上一页的页码，检查连续性
        if last_page_number is not None:
            for candidate in page_candidates:
                if abs(candidate[0] - (last_page_number + 1)) <= 1:  # 允许±1的误差
                    print(f"   ✅ 基于连续性选择: {candidate[0]}")
                    return str(candidate[0])
        
        # 否则使用得分最高的候选（降低阈值，因为我们已经限制了区域）
        if best_candidate[1] >= 3:
            return str(best_candidate[0])
    
    print(f"   ⚠️ 页脚区域未找到可靠页码")
    return fallback_page_number(current_page_index, total_pages, last_page_number)


def fallback_page_number(current_page_index, total_pages, last_page_number):
    """
    页码识别失败时的后备方案
    """
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


# 为了向后兼容，保留原函数名
def extract_page_number_improved(ocr_results, height, width, current_page_index, total_pages, last_page_number):
    """
    向后兼容的函数名，调用新的页脚识别函数
    """
    return extract_page_number_from_footer(ocr_results, height, width, current_page_index, total_pages, last_page_number)