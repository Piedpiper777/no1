import os
import re
import cv2
import numpy as np
import pdfplumber
from PIL import Image, ImageEnhance
from paddleocr import PPStructure, PaddleOCR
import fitz  # PyMuPDF

def detect_pdf_type(pdf_path, sample_pages=3):
    """
    检测PDF类型：文本型 vs 图片型
    
    Args:
        pdf_path: PDF文件路径
        sample_pages: 采样页数进行检测
    
    Returns:
        'text': 文本型PDF
        'image': 图片型PDF
        'mixed': 混合型PDF
    """
    print("🔍 正在分析PDF类型...")
    
    text_pages = 0
    image_pages = 0
    total_checked = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        # 检查前几页来判断类型
        pages_to_check = min(sample_pages, len(pdf.pages))
        
        for i in range(pages_to_check):
            page = pdf.pages[i]
            text = page.extract_text()
            
            total_checked += 1
            
            # 判断标准：
            # 1. 文本长度
            # 2. 中文字符比例
            # 3. 有效文本行数
            if text and len(text.strip()) > 50:
                # 计算中文字符比例
                chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
                total_chars = len(re.sub(r'\s', '', text))
                
                if total_chars > 0:
                    chinese_ratio = chinese_chars / total_chars
                    # 如果中文字符占比>10%，认为是有效文本页
                    if chinese_ratio > 0.1 or total_chars > 200:
                        text_pages += 1
                        continue
            
            # 如果文本提取失败或文本很少，判断为图片页
            image_pages += 1
    
    # 判断逻辑
    text_ratio = text_pages / total_checked
    
    if text_ratio >= 0.8:
        pdf_type = 'text'
        print(f"📄 检测结果: 文本型PDF (文本页: {text_pages}/{total_checked})")
    elif text_ratio <= 0.2:
        pdf_type = 'image'
        print(f"🖼️  检测结果: 图片型PDF (图片页: {image_pages}/{total_checked})")
    else:
        pdf_type = 'mixed'
        print(f"📄🖼️  检测结果: 混合型PDF (文本页: {text_pages}, 图片页: {image_pages})")
    
    return pdf_type

def detect_content_area(page, margin_ratio=0.05):
    """
    改进版动态检测页面内容区域,自动过滤页眉页脚和页码
    
    Args:
        page: pdfplumber页面对象
        margin_ratio: 基础边距比例（默认5%）
    
    Returns:
        tuple: (x0, y0, x1, y1) 内容区域坐标
    """
    width, height = page.width, page.height
    
    # 默认边距
    default_margin_x = width * margin_ratio
    default_margin_y = height * margin_ratio
    
    try:
        # 获取页面上所有文本对象
        chars = page.chars
        
        if not chars:
            return (
                default_margin_x,
                default_margin_y,
                width - default_margin_x,
                height - default_margin_y
            )
            
        # 按y坐标分组统计字符分布
        y_groups = {}
        for char in chars:
            y = int(char['top'])
            if y not in y_groups:
                y_groups[y] = []
            y_groups[y].append(char)
        
        # 页码检测特征
        def is_page_number(char_group):
            # 1. 长度特征：页码通常很短
            if len(char_group) > 5:
                return False
                
            # 2. 数字特征：页码通常是纯数字
            text = ''.join(char['text'] for char in char_group)
            if not text.isdigit():
                return False
                
            # 3. 位置特征：通常在页面底部居中
            avg_x = sum(char['x0'] for char in char_group) / len(char_group)
            center_zone = (width * 0.4, width * 0.6)  # 中间区域
            if not (center_zone[0] < avg_x < center_zone[1]):
                return False
                
            return True

        # 分析垂直方向的文本密度
        density_threshold = len(chars) / height * 0.3  # 动态密度阈值
        
        # 找出页眉页脚的边界
        header_bottom = 0
        footer_top = height
        
        sorted_y = sorted(y_groups.keys())
        last_valid_text_y = 0  # 记录最后一个有效文本的位置
        
        # 检测页眉
        for y in sorted_y:
            if len(y_groups[y]) < density_threshold and not is_page_number(y_groups[y]):
                header_bottom = y
            else:
                break

        # 检测页脚(自下而上)
        for y in reversed(sorted_y):
            # 如果是页码，跳过这一行
            if is_page_number(y_groups[y]):
                continue
                
            if len(y_groups[y]) < density_threshold:
                footer_top = y
            else:
                last_valid_text_y = y
                break
        
        # 获取水平方向边界
        x_coords = [char['x0'] for char in chars] + [char['x1'] for char in chars]
        text_left = max(min(x_coords), default_margin_x)
        text_right = min(max(x_coords), width - default_margin_x)
        
        # 添加安全边距
        safe_margin = min(width, height) * 0.02
        content_x0 = max(0, text_left - safe_margin)
        content_y0 = max(header_bottom + safe_margin, default_margin_y)
        content_x1 = min(width, text_right + safe_margin)
        
        # 使用最后一个有效文本位置来设置底部边界
        if last_valid_text_y:
            content_y1 = min(last_valid_text_y + safe_margin, height - default_margin_y)
        else:
            content_y1 = min(footer_top - safe_margin, height - default_margin_y)
        
        # 确保坐标有效
        content_x0 = max(0, content_x0)
        content_y0 = max(0, content_y0)
        content_x1 = min(width, content_x1)
        content_y1 = min(height, content_y1)
        
        return (content_x0, content_y0, content_x1, content_y1)
        
    except Exception as e:
        print(f"⚠️ 内容区域检测失败: {str(e)}")
        # 如果分析失败,使用默认边距
        return (
            default_margin_x,
            default_margin_y,
            width - default_margin_x,
            height - default_margin_y
        )

def fix_chinese_soft_breaks(s):
    """修复中文换行导致的拆词问题"""
    s = re.sub(r'-\s+', '', s)
    s = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', s)
    return s

def preprocess_image(img):
    """图片预处理，提高OCR识别率"""
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 图片增强
    # 1. 高斯去噪
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. 对比度增强
    pil_img = Image.fromarray(denoised)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(1.5)  # 增强对比度
    
    # 3. 锐化
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    sharpened = sharpness_enhancer.enhance(1.2)
    
    # 转换回opencv格式
    processed_img = np.array(sharpened)
    
    return processed_img

def process_text_with_paragraphs(text_lines):
    """修改版：仅处理带括号的编号"""
    text_output = []
    current_para = []
    merged_lines = []

    # 第一阶段：智能合并被拆分的段落行（保持原逻辑）
    buffer = ""
    for line in (l.strip() for l in text_lines if l.strip()):
        # 仅检测带括号的编号作为新段落起始
        if re.match(r'^[\[\(\【（［]+.*\d+.*[\]）\】］]*', line):  # 修改行合并条件
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
        buffer = f"{buffer} {line}".strip() if buffer else line
    
    if buffer:
        merged_lines.append(buffer)

    # 第二阶段：精确匹配带括号的编号
    para_pattern = re.compile(
        r'^('
        r'[\[\(\【（\［]+\d+[\]）\】］]*'  # 有前括号
        r'|'                             # 或 
        r'\d+[\]）\】\］]+'               # 有后括号
        r')'
        r'[\.\。]?'          # 可选结束符
        r'\s*'               # 后续空格
    )

    for line in merged_lines:
        

        # 仅匹配带括号的编号
        match = para_pattern.match(line)
        if match:
            # 计算编号部分长度（保持原逻辑）
            match_len = match.end()
            remaining = line[match_len:].strip()

            # 保存上一个段落（保持原逻辑）
            if current_para:
                joined = fix_chinese_soft_breaks(" ".join(current_para))
                text_output.append(joined)
                current_para = []
            
            if remaining:
                current_para.append(remaining)
        else:
            # 无编号内容处理（保持原逻辑）
            if current_para:
                current_para.append(line)
            else:
                cleaned = fix_chinese_soft_breaks(line)
                text_output.append(cleaned)

    # 处理最后一段（保持原逻辑）
    if current_para:
        joined = fix_chinese_soft_breaks(" ".join(current_para))
        text_output.append(joined)

    return text_output

def extract_images_tables_with_ppstructure(pdf_path, page_num, current_id, img_counter, table_counter, img_dir, para_buffer):
    """使用PPStructure提取单页的图片和表格"""
    
    structure_engine = PPStructure(
        recovery=False,
        lang='ch',
        show_log=False
    )
    
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_num]
    
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    try:
        result = structure_engine(img)
        result.sort(key=lambda x: x['bbox'][1])
        
        for item in result:
            bbox = item['bbox']
            item_type = item['type']
            
            if item_type == 'figure':
                img_counter += 1
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                cropped_img = img[y0:y1, x0:x1]
                img_name = f"{current_id}_{img_counter}.png" if current_id else f"page{page_num+1}_img{img_counter}.png"
                img_path = os.path.join(img_dir, img_name)
                cv2.imwrite(img_path, cropped_img)
                para_buffer += f"\n[IMG_{img_counter}]"
                print(f"📷 提取图片: {img_name}")
            
            elif item_type == 'table':
                table_counter += 1
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                cropped_table = img[y0:y1, x0:x1]
                table_name = f"{current_id}_table{table_counter}.png" if current_id else f"page{page_num+1}_table{table_counter}.png"
                table_path = os.path.join(img_dir, table_name)
                cv2.imwrite(table_path, cropped_table)
                para_buffer += f"\n[TABLE_{table_counter}]"
                print(f"📊 提取表格: {table_name}")
    
    except Exception as e:
        print(f"⚠️ PPStructure处理第{page_num+1}页时出错: {str(e)}")
    
    finally:
        pdf_document.close()
    
    return img_counter, table_counter, para_buffer

def extract_text_pdf(pdf_path, output_dir):
    """
    文本型PDF处理器
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
    """
    print("📄 使用文本型PDF处理模式...")
    
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    text_output = []
    img_counter = 0
    table_counter = 0
    all_text_lines = []  # 收集所有文本行
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"处理第 {page_num + 1}/{len(pdf.pages)} 页")
            
            # 动态检测内容区域
            content_area = detect_content_area(page)
            crop = page.within_bbox(content_area)
            text = crop.extract_text()
            
            if not text:
                print(f"  ⚠️ 第{page_num+1}页未提取到文本")
                continue
            
            # 按行分割并清理
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            all_text_lines.extend(lines)
            
            print(f"  📝 提取了 {len(lines)} 行文本")
            
            # PPStructure增强图片表格提取
            try:
                temp_para_buffer = ""
                img_counter, table_counter, _ = extract_images_tables_with_ppstructure(
                    pdf_path, page_num, None, img_counter, table_counter, img_dir, temp_para_buffer
                )
            except Exception as e:
                print(f"⚠️ 页面{page_num+1}的PPStructure处理失败: {str(e)}")
    
    # 智能处理文本段落
    print(f"🔄 处理提取的文本，共 {len(all_text_lines)} 行")
    
    # 使用改进的段落分割逻辑
    text_output = smart_paragraph_split(all_text_lines)
    
    # 保存文本文件
    text_file = os.path.join(output_dir, "descriptions.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))
    
    # 生成JSON文件
    json_file = os.path.join(output_dir, "descriptions.json")
    try:
        convert_text_to_json(text_file, json_file)
        print(f"✅ JSON文件已生成: {json_file}")
    except Exception as e:
        print(f"⚠️ JSON转换失败: {str(e)}")
    
    print(f"✅ 文本提取完成，共 {len(text_output)} 个段落")
    return len(text_output), img_counter, table_counter

import re

def smart_paragraph_split(text_lines):
    paragraphs = []
    current_paragraph = []

    # 常见编号格式
    number_patterns = [
        r'^\d+[\.\．]',                    # 1. 2. 3.
        r'^[\[\(（\（]+\d+[\]\)）\）]+',    # [1] (1) （1）
        r'^\d+[\)）]',                    # 1) 2)
    ]

    # 结构性标题关键词
    SECTION_TITLES = {
        "技术领域": ["技术领域"],
        "背景技术": ["背景技术"],
        "发明内容": ["发明内容", "实用新型内容"],
        "附图说明": ["附图说明"],
        "具体实施方式": ["具体实施方式"]
    }

    def is_numbered_line(line):
        for pattern in number_patterns:
            if re.match(pattern, line):
                return True
        return False

    def is_title_like(line):
        return len(line) < 50 and (line.isupper() or line.endswith('：') or line.endswith(':'))

    def is_section_title(line):
        stripped = line.strip().replace(" ", "")
        for section, aliases in SECTION_TITLES.items():
            for alias in aliases:
                if stripped == alias:
                    return section
        return None

    def fix_chinese_soft_breaks(s):
        s = re.sub(r'-\s+', '', s)
        s = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', s)
        return s

    for i, line in enumerate(text_lines):
        line = line.strip()
        if not line:
            continue

        section = is_section_title(line)
        if section:
            # 保存当前段落
            if current_paragraph:
                paragraph_text = fix_chinese_soft_breaks(" ".join(current_paragraph))
                if paragraph_text.strip():
                    paragraphs.append(paragraph_text)
                current_paragraph = []

            # 小标题独立成段
            paragraphs.append(section)
            continue

        is_new_paragraph = False
        if is_numbered_line(line):
            is_new_paragraph = True
        elif is_title_like(line):
            is_new_paragraph = True
        elif i > 0 and len(line) < 20 and not line.endswith(('，', '。', '；', '：', ',', '.', ';', ':')):
            is_new_paragraph = True

        if is_new_paragraph and current_paragraph:
            paragraph_text = fix_chinese_soft_breaks(" ".join(current_paragraph))
            if paragraph_text.strip():
                paragraphs.append(paragraph_text)
            current_paragraph = []

        current_paragraph.append(line)

    if current_paragraph:
        paragraph_text = fix_chinese_soft_breaks(" ".join(current_paragraph))
        if paragraph_text.strip():
            paragraphs.append(paragraph_text)

    # 过滤过短的段落（可能是噪声）
    filtered_paragraphs = [p for p in paragraphs if len(p.strip()) > 1]

    return filtered_paragraphs

def extract_image_pdf(pdf_path, output_dir):
    """
    图片型PDF处理器
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
    """
    print(f"🖼️ 使用图片型PDF处理模式: {os.path.basename(pdf_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # 初始化OCR引擎
    print("🔧 初始化OCR引擎...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,
        show_log=False
    )
    
    # 初始化结构分析引擎
    print("🔧 初始化结构分析引擎...")
    structure_engine = PPStructure(
        recovery=True,
        lang='ch',
        show_log=False
    )
    
    # 打开PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    
    all_text_lines = []  # 收集所有文本行
    img_counter = 0
    table_counter = 0
    
    print(f"📖 PDF总页数: {total_pages}")
    
    for page_num in range(total_pages):
        print(f"\n--- 📄 处理第 {page_num + 1}/{total_pages} 页 ---")
        
        page = pdf_document[page_num]
        
        # 转换为高分辨率图片
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 图片预处理
        processed_img = preprocess_image(original_img)
        
        page_text_lines = []
        structure_success = False
        
        # 方法1: 尝试使用PPStructure进行结构化分析
        print("🔬 尝试PPStructure结构化分析...")
        try:
            structure_result = structure_engine(original_img)
            
            if structure_result:
                # 按y坐标排序，保证阅读顺序
                structure_result.sort(key=lambda x: x['bbox'][1])
                
                for item in structure_result:
                    bbox = item['bbox']
                    item_type = item['type']
                    
                    if item_type == 'text':
                        # 处理文本区域
                        text_content = item.get('res', [])
                        if isinstance(text_content, list) and text_content:
                            for text_item in text_content:
                                if isinstance(text_item, dict) and 'text' in text_item:
                                    confidence = text_item.get('confidence', 0)
                                    if confidence > 0.5:
                                        page_text_lines.append(text_item['text'])
                                        structure_success = True
                                elif isinstance(text_item, str):
                                    page_text_lines.append(text_item)
                                    structure_success = True
                    
                    elif item_type == 'figure':
                        # 处理图片
                        img_counter += 1
                        x0, y0, x1, y1 = [int(coord) for coord in bbox]
                        
                        h, w = original_img.shape[:2]
                        x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
                        y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
                        
                        if x1 > x0 and y1 > y0:
                            cropped_img = original_img[y0:y1, x0:x1]
                            img_name = f"page{page_num+1}_img{img_counter}.png"
                            img_path = os.path.join(img_dir, img_name)
                            cv2.imwrite(img_path, cropped_img)
                            page_text_lines.append(f"[IMG_{img_counter}]")
                            print(f"  📷 提取图片: {img_name}")
                    
                    elif item_type == 'table':
                        # 处理表格
                        table_counter += 1
                        x0, y0, x1, y1 = [int(coord) for coord in bbox]
                        
                        h, w = original_img.shape[:2]
                        x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
                        y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
                        
                        if x1 > x0 and y1 > y0:
                            cropped_table = original_img[y0:y1, x0:x1]
                            table_name = f"page{page_num+1}_table{table_counter}.png"
                            table_path = os.path.join(img_dir, table_name)
                            cv2.imwrite(table_path, cropped_table)
                            page_text_lines.append(f"[TABLE_{table_counter}]")
                            print(f"  📊 提取表格: {table_name}")
        
        except Exception as e:
            print(f"⚠️ PPStructure分析失败: {str(e)}")
        
        # 方法2: 如果PPStructure没有成功提取文本，使用纯OCR
        if not structure_success:
            print("🔤 使用纯OCR模式...")
            try:
                # 在OCR之前添加内容区域检测和裁剪
                page = pdf_document[page_num]
                content_area = detect_content_area(page)
                x0, y0, x1, y1 = [int(coord) for coord in content_area]

                # 裁剪图片到内容区域
                h, w = original_img.shape[:2]
                content_img = original_img[
                    int(y0 * h / page.height):int(y1 * h / page.height),
                    int(x0 * w / page.width):int(x1 * w / page.width)
                ]

                # 对裁剪后的图片进行OCR处理
                processed_img = preprocess_image(content_img)
                ocr_result = ocr_engine.ocr(processed_img, cls=True)
                
                if ocr_result and ocr_result[0]:
                    # 按照y坐标排序OCR结果
                    ocr_lines = sorted(ocr_result[0], key=lambda x: x[0][0][1])
                    
                    for line in ocr_lines:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # 置信度过滤
                        if confidence > 0.6:
                            page_text_lines.append(text)
            
            except Exception as e:
                print(f"❌ OCR处理失败: {str(e)}")
        
        # 将页面文本添加到总文本中
        if page_text_lines:
            all_text_lines.extend(page_text_lines)
        else:
            print("  ⚠️ 本页未提取到文本")
    
    pdf_document.close()
    
    # 智能处理文本段落
    print(f"\n🔄 处理提取的文本，共 {len(all_text_lines)} 行")
    text_output = smart_paragraph_split(all_text_lines)
    
    # 保存文本文件
    text_file = os.path.join(output_dir, "descriptions.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))
    
    # 生成JSON文件
    json_file = os.path.join(output_dir, "descriptions.json")
    try:
        convert_text_to_json(text_file, json_file)
        print(f"✅ JSON文件已生成: {json_file}")
    except Exception as e:
        print(f"⚠️ JSON转换失败: {str(e)}")
    
    return len(text_output), img_counter, table_counter

def convert_text_to_json(text_file, json_file):
    """
    将提取的文本转换为结构化JSON格式
    
    Args:
        text_file: 输入的txt文件路径
        json_file: 输出的json文件路径
    """
    import json
    
    # 定义所有可能的部分标题
    SECTIONS = {
        "技术领域": ["技术领域"],
        "背景技术": ["背景技术"],
        "发明内容": ["发明内容", "实用新型内容"],
        "附图说明": ["附图说明"],
        "具体实施方式": ["具体实施方式", "具体实施例"]
    }
    
    # 初始化结构
    result = {
        "题目": [],
        "技术领域": [],
        "背景技术": [],
        "发明内容/实用新型内容": [],
        "附图说明": [],
        "具体实施方式": []
    }
    
    # 读取文本文件
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 提取题目 (第一行)
    if lines:
        result["题目"].append(lines[0])
        lines = lines[1:]  # 移除题目行
    
    # 查找各个部分的起始位置
    section_positions = {}
    current_section = None
    
    for i, line in enumerate(lines):
        stripped_line = line.strip().replace(" ", "")
        
        # 检查是否是小标题
        for section, aliases in SECTIONS.items():
            if stripped_line in aliases:
                section_positions[i] = section
                current_section = section
                break
    
    # 按顺序提取各个部分的内容
    if section_positions:
        # 将位置信息转换为排序后的列表
        sorted_positions = sorted(section_positions.items())
        
        # 处理各个部分
        for i, (pos, section) in enumerate(sorted_positions):
            start = pos + 1  # 跳过标题行
            
            # 确定结束位置
            if i < len(sorted_positions) - 1:
                end = sorted_positions[i + 1][0]
            else:
                end = len(lines)
            
            # 提取当前部分的内容
            content = lines[start:end]
            result[section].extend([line for line in content if line.strip()])
    
    # 写入JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

# 在 smart_extract_pdf 函数的最后添加 JSON 转换
def smart_extract_pdf(pdf_path, output_dir):
    """智能PDF提取 - 自动判断类型并选择合适的处理方法"""
    
    print(f"🚀 开始智能处理PDF: {os.path.basename(pdf_path)}")
    
    # 第一步：检测PDF类型
    pdf_type = detect_pdf_type(pdf_path)
    
    # 第二步：选择对应的处理方法
    if pdf_type == 'text':
        paragraphs, images, tables = extract_text_pdf(pdf_path, output_dir)  # 使用修复版
    elif pdf_type == 'image':
        paragraphs, images, tables = extract_image_pdf(pdf_path, output_dir)
    else:  # mixed
        print("📄🖼️ 混合型PDF，使用修复版文本模式处理（主要逻辑）+ OCR补充")
        # 混合型使用修复版文本模式，后续可以优化为逐页判断
        paragraphs, images, tables = extract_text_pdf(pdf_path, output_dir)
        
        print(f"\n✅ 处理完成！")
        print(f"   📊 PDF类型: {pdf_type}")
        print(f"   📄 段落数: {paragraphs}")
        print(f"   📷 图片数: {images}")
        
        # 修正：使用正确的文件名
        text_file = os.path.join(output_dir, "descriptions.txt")  # 修改这里
        json_file = os.path.join(output_dir, "descriptions.json") # 修改这里
    
    try:
        json_content = convert_text_to_json(text_file, json_file)
        print(f"   📋 JSON文件已生成: {json_file}")
    except Exception as e:
        print(f"⚠️ JSON转换失败: {str(e)}")
    
    return {
        'pdf_type': pdf_type,
        'paragraphs': paragraphs,
        'images': images,
        'tables': tables,
        'json_file': json_file if 'json_file' in locals() else None
    }

# 用法示例
if __name__ == "__main__":
    pdf_path = r"/workspace/project/output/descriptions.pdf"
    output_dir = "output_descriptions"
    
    try:
        result = smart_extract_pdf(pdf_path, output_dir)
        
        print(f"\n🎉 全部完成！输出文件：")
        print(f"   📄 descriptions.txt - 结构化文本 ({result['paragraphs']} 段落)")  # 修改这里
        print(f"   📁 images/ - {result['images']} 个图片 + {result['tables']} 个表格")
        print(f"   🔍 PDF类型: {result['pdf_type']}")
        
        # 如果有JSON文件，显示路径
        if result.get('json_file'):
            print(f"   📋 JSON文件: {result['json_file']}")
        
    except ImportError as e:
        print(f"❌ 依赖库缺失: {str(e)}")
        print("💡 请安装: pip install paddlepaddle paddleocr PyMuPDF opencv-python pdfplumber pillow")