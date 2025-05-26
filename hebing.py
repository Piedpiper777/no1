import os
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR, PPStructure
import fitz  # PyMuPDF

def detect_pdf_type(pdf_path, sample_pages=5):
    """
    检测PDF类型：文本型、图片型或混合型
    通过分析前几页的文本密度和中文比例来判断
    
    Args:
        pdf_path: PDF文件路径
        sample_pages: 用于分析的样本页数
    
    Returns:
        'text': 文本型PDF（文本页比例 >= 80%）
        'image': 图片型PDF（文本页比例 <= 20%）
        'mixed': 混合型PDF（其他情况）
    """
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    sample_pages = min(sample_pages, total_pages)
    
    text_page_count = 0
    
    for page_num in range(sample_pages):
        page = pdf_document[page_num]
        text = page.get_text()
        
        # 计算文本长度
        text_length = len(text)
        
        # 计算中文字符比例
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        chinese_ratio = len(chinese_chars) / text_length if text_length > 0 else 0
        
        # 计算有效文本行数（至少包含3个字符的行）
        valid_lines = [line for line in text.split('\n') if len(line.strip()) >= 3]
        line_count = len(valid_lines)
        
        # 判断是否为文本页的条件
        is_text_page = (text_length > 500) and (chinese_ratio > 0.3) and (line_count > 10)
        
        if is_text_page:
            text_page_count += 1
    
    pdf_document.close()
    
    # 计算文本页比例
    text_page_ratio = text_page_count / sample_pages
    
    # 根据比例判断PDF类型
    if text_page_ratio >= 0.8:
        return 'text'
    elif text_page_ratio <= 0.2:
        return 'image'
    else:
        return 'mixed'

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

def process_text_with_paragraphs(text_lines, debug=False):
    """改进版段落处理：兼容括号不成对的编号并统一删除"""
    text_output = []
    current_para = []
    merged_lines = []

    # 第一阶段：合并被拆分的段落内容
    temp_line = ""
    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # 检测是否为新段落起始行（宽松匹配）
        is_new_para = re.match(r'^[\[\(\【（]*\d+[\]）\】]*', stripped) \
                   or re.match(r'^\d+[\.\。]', stripped) \
                   or re.match(r'^第\d+[段节]', stripped)
        
        if is_new_para and temp_line:
            merged_lines.append(temp_line.strip())
            temp_line = stripped
        else:
            temp_line += " " + stripped if temp_line else stripped
    
    if temp_line:
        merged_lines.append(temp_line.strip())

    # 第二阶段：处理合并后的文本行
    for line in merged_lines:
        if debug:
            print(f"处理文本行: {line[:50]}...")

        # 匹配所有可能的段落编号模式
        para_pattern = re.compile(
            r'^(?:[\[\［\(\【（]*)'      # 可能的前括号（0或多个）
            r'(\d+)'                 # 数字编号
            r'[\]）\】\］]*'            # 可能的后括号（0或多个）
            r'[\.\。]?'              # 可能的标点
            r'(?:段|节)?'            # 可能的后缀
            r'\s*'                   # 可能的空格
        )

        # 尝试提取编号部分
        match = para_pattern.match(line)
        if match:
            # 提取编号并计算匹配部分的长度
            number_part = match.group(1)
            match_length = match.end()
            
            # 保存上一个段落
            if current_para:
                cleaned_para = fix_chinese_soft_breaks(" ".join(current_para).strip())
                if cleaned_para:
                    text_output.append(cleaned_para)
                    if debug:
                        print(f"完成段落: {cleaned_para[:50]}...")
            
            # 开始新段落（去除编号部分）
            remaining_text = line[match_length:].strip()
            current_para = [remaining_text] if remaining_text else []
        else:
            if current_para:
                current_para.append(line)
            else:
                # 无编号的独立文本
                cleaned_line = fix_chinese_soft_breaks(line)
                text_output.append(cleaned_line)
                if debug:
                    print(f"独立文本: {cleaned_line[:50]}...")

    # 处理最后一个段落
    if current_para:
        cleaned_para = fix_chinese_soft_breaks(" ".join(current_para).strip())
        if cleaned_para:
            text_output.append(cleaned_para)

    return text_output

def extract_images_tables_with_ppstructure(img_path, output_dir, page_num, table_engine=None):
    """
    使用PPStructure从PDF单页中提取图片和表格
    
    Args:
        img_path: 图片路径
        output_dir: 输出目录
        page_num: 页码
        table_engine: 表格识别引擎实例
    
    Returns:
        text: 提取的文本
        img_counter: 提取的图片数量
        table_counter: 提取的表格数量
    """
    if table_engine is None:
        # 创建表格识别引擎实例
        table_engine = PPStructure(show_log=False, lang="ch")
    
    # 读取图片
    img = cv2.imread(img_path)
    
    # 使用PPStructure进行版面分析
    result = table_engine(img)
    
    # 准备输出文本
    text = []
    img_counter = 0
    table_counter = 0
    
    # 创建图片和表格的输出目录
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # 处理识别结果
    for region in result:
        if region['type'] == 'text':
            # 处理文本区域
            text_region = region['res']
            text.append(text_region['text'])
        elif region['type'] == 'table':
            # 处理表格区域
            table_region = region['res']
            table_img = table_region['img']
            
            # 保存表格图片
            table_img_path = os.path.join(tables_dir, f"table_p{page_num}_{table_counter}.png")
            cv2.imwrite(table_img_path, table_img)
            
            # 在文本中插入表格引用
            text.append(f"\n\n[表格 {page_num}-{table_counter}] 见 {os.path.basename(table_img_path)}\n\n")
            
            table_counter += 1
        elif region['type'] == 'figure':
            # 处理图片区域
            figure_region = region['res']
            figure_img = figure_region['img']
            
            # 保存图片
            figure_img_path = os.path.join(images_dir, f"fig_p{page_num}_{img_counter}.png")
            cv2.imwrite(figure_img_path, figure_img)
            
            # 在文本中插入图片引用
            text.append(f"\n\n[图片 {page_num}-{img_counter}] 见 {os.path.basename(figure_img_path)}\n\n")
            
            img_counter += 1
    
    # 合并所有文本
    full_text = "\n".join(text)
    
    return full_text, img_counter, table_counter

def extract_text_pdf(pdf_path, output_dir, debug=True):
    """
    处理文本型PDF文件
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        debug: 是否输出调试信息
    
    Returns:
        处理结果统计信息
    """
    print(f"📝 开始处理文本型PDF: {os.path.basename(pdf_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # 打开PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    
    # 创建表格识别引擎
    table_engine = PPStructure(show_log=False, lang="ch")
    
    all_text_lines = []  # 收集所有文本行
    total_img_count = 0
    total_table_count = 0
    
    print(f"📖 PDF总页数: {total_pages}")
    
    for page_num in range(total_pages):
        print(f"\n--- 📄 处理第 {page_num + 1}/{total_pages} 页 ---")
        
        page = pdf_document[page_num]
        
        # 提取文本
        text = page.get_text()
        page_text_lines = text.split('\n')
        all_text_lines.extend(page_text_lines)
        
        # 提取图片和表格
        mat = fitz.Matrix(2.0, 2.0)  # 提高图片分辨率
        pix = page.get_pixmap(matrix=mat)
        
        # 保存页面图片用于提取表格和图片
        page_img_path = os.path.join(debug_dir, f"page_{page_num+1}.png") if debug else f"temp_page_{page_num+1}.png"
        pix.save(page_img_path)
        
        # 使用PPStructure提取图片和表格
        page_text, img_count, table_count = extract_images_tables_with_ppstructure(
            page_img_path, output_dir, page_num+1, table_engine
        )
        
        total_img_count += img_count
        total_table_count += table_count
        
        # 如果提取到了额外的文本，添加到文本行中
        if page_text:
            page_text_lines = page_text.split('\n')
            all_text_lines.extend(page_text_lines)
        
        # 删除临时文件
        if not debug and os.path.exists(page_img_path):
            os.remove(page_img_path)
    
    pdf_document.close()
    
    # 处理文本，识别段落
    print(f"\n🔄 处理提取的文本，共 {len(all_text_lines)} 行")
    text_output = process_text_with_paragraphs(all_text_lines, debug=debug)
    
    # 写入文本文件
    text_file = os.path.join(output_dir, "text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))  # 段落间用双换行分隔
    
    # 同时保存原始提取的文本（调试用）
    if debug:
        raw_text_file = os.path.join(debug_dir, "raw_text.txt")
        with open(raw_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text_lines))
        print(f"🔍 原始文本已保存: {raw_text_file}")
    
    # 输出统计信息
    print(f"\n🎉 文本型PDF处理完成！")
    print(f"   📄 总页数: {total_pages}")
    print(f"   📝 原始文本行数: {len(all_text_lines)}")
    print(f"   📝 处理后段落数: {len(text_output)}")
    print(f"   📷 图片数: {total_img_count}")
    print(f"   📊 表格数: {total_table_count}")
    print(f"   📁 输出目录: {output_dir}")
    
    if debug:
        print(f"   🔍 调试文件: {os.path.join(output_dir, 'debug')}")
    
    return {
        'pages': total_pages,
        'raw_lines': len(all_text_lines),
        'paragraphs': len(text_output),
        'images': total_img_count,
        'tables': total_table_count
    }

# 新的图片型PDF处理函数
def extract_image_pdf(pdf_path, output_dir, debug=True):
    """
    处理图片型PDF文件（集成新的处理逻辑）
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        debug: 是否输出调试信息
    
    Returns:
        处理结果统计信息
    """
    print(f"🖼️ 开始处理图片型PDF: {os.path.basename(pdf_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # 打开PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    
    # 创建OCR和表格识别引擎
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    table_engine = PPStructure(show_log=False, lang="ch")
    
    all_text_lines = []  # 收集所有文本行
    total_img_count = 0
    total_table_count = 0
    
    print(f"📖 PDF总页数: {total_pages}")
    
    for page_num in range(total_pages):
        print(f"\n--- 📄 处理第 {page_num + 1}/{total_pages} 页 ---")
        
        page = pdf_document[page_num]
        
        # 转换为高分辨率图片
        mat = fitz.Matrix(3.0, 3.0)  # 提高分辨率以获得更好的OCR效果
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if debug:
            debug_img_path = os.path.join(debug_dir, f"page_{page_num+1}_original.png")
            cv2.imwrite(debug_img_path, original_img)
            print(f"🔍 原始图片已保存: {debug_img_path}")
        
        # 图片预处理
        processed_img = preprocess_image(original_img)
        
        if debug:
            processed_img_path = os.path.join(debug_dir, f"page_{page_num+1}_processed.png")
            cv2.imwrite(processed_img_path, processed_img)
            print(f"🔍 预处理后图片已保存: {processed_img_path}")
        
        # 使用PPStructure进行版面分析和内容提取
        try:
            result = table_engine(processed_img)
            
            page_text_lines = []
            page_img_count = 0
            page_table_count = 0
            
            # 处理识别结果
            for region in result:
                if region['type'] == 'text':
                    # 处理文本区域
                    text_region = region['res']
                    text = text_region['text']
                    page_text_lines.append(text)
                elif region['type'] == 'table':
                    # 处理表格区域
                    table_region = region['res']
                    table_img = table_region['img']
                    
                    # 保存表格图片
                    table_img_path = os.path.join(img_dir, f"table_p{page_num+1}_{page_table_count}.png")
                    cv2.imwrite(table_img_path, table_img)
                    
                    # 在文本中插入表格引用
                    page_text_lines.append(f"\n\n[表格 {page_num+1}-{page_table_count}] 见 {os.path.basename(table_img_path)}\n\n")
                    
                    page_table_count += 1
                elif region['type'] == 'figure':
                    # 处理图片区域
                    figure_region = region['res']
                    figure_img = figure_region['img']
                    
                    # 保存图片
                    figure_img_path = os.path.join(img_dir, f"fig_p{page_num+1}_{page_img_count}.png")
                    cv2.imwrite(figure_img_path, figure_img)
                    
                    # 在文本中插入图片引用
                    page_text_lines.append(f"\n\n[图片 {page_num+1}-{page_img_count}] 见 {os.path.basename(figure_img_path)}\n\n")
                    
                    page_img_count += 1
            
            total_img_count += page_img_count
            total_table_count += page_table_count
            
            print(f"✅ 从第 {page_num+1} 页提取了 {len(page_text_lines)} 行文本，{page_img_count} 张图片，{page_table_count} 个表格")
            
        except Exception as e:
            print(f"⚠️ PPStructure处理失败: {str(e)}，降级使用纯OCR")
            
            # 降级使用纯OCR
            result = ocr.ocr(processed_img, cls=True)
            page_text_lines = []
            
            for line in result[0]:
                text = line[1][0]
                page_text_lines.append(text)
            
            print(f"✅ 从第 {page_num+1} 页提取了 {len(page_text_lines)} 行文本（纯OCR）")
        
        all_text_lines.extend(page_text_lines)
    
    pdf_document.close()
    
    # 处理文本，识别段落
    print(f"\n🔄 处理提取的文本，共 {len(all_text_lines)} 行")
    text_output = process_text_with_paragraphs(all_text_lines, debug=debug)
    
    # 写入文本文件
    text_file = os.path.join(output_dir, "text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))  # 段落间用双换行分隔
    
    # 同时保存原始提取的文本（调试用）
    if debug:
        raw_text_file = os.path.join(debug_dir, "raw_text.txt")
        with open(raw_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text_lines))
        print(f"🔍 原始文本已保存: {raw_text_file}")
    
    # 输出统计信息
    print(f"\n🎉 图片型PDF处理完成！")
    print(f"   📄 总页数: {total_pages}")
    print(f"   📝 原始文本行数: {len(all_text_lines)}")
    print(f"   📝 处理后段落数: {len(text_output)}")
    print(f"   📷 图片数: {total_img_count}")
    print(f"   📊 表格数: {total_table_count}")
    print(f"   📁 输出目录: {output_dir}")
    
    if debug:
        print(f"   🔍 调试文件: {os.path.join(output_dir, 'debug')}")
    
    return {
        'pages': total_pages,
        'raw_lines': len(all_text_lines),
        'paragraphs': len(text_output),
        'images': total_img_count,
        'tables': total_table_count
    }

def smart_extract_pdf(pdf_path, output_dir="output", debug=True):
    """
    智能提取PDF内容，根据PDF类型选择不同的处理策略
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        debug: 是否输出调试信息
    
    Returns:
        处理结果统计信息
    """
    print(f"🚀 开始智能处理PDF: {os.path.basename(pdf_path)}")
    
    # 检测PDF类型
    pdf_type = detect_pdf_type(pdf_path)
    print(f"🔍 检测到PDF类型: {pdf_type}")
    
    # 根据PDF类型选择处理方法
    if pdf_type == 'text':
        output_subdir = os.path.join(output_dir, "text_pdf")
        result = extract_text_pdf(pdf_path, output_subdir, debug)
    elif pdf_type == 'image':
        output_subdir = os.path.join(output_dir, "image_pdf")
        result = extract_image_pdf(pdf_path, output_subdir, debug)
    else:  # mixed
        output_subdir = os.path.join(output_dir, "mixed_pdf")
        # 对于混合型PDF，使用图片型PDF处理方式
        result = extract_image_pdf(pdf_path, output_subdir, debug)
    
    # 添加PDF类型到结果中
    result['pdf_type'] = pdf_type
    
    print(f"\n📊 处理结果汇总:")
    print(f"   📄 PDF类型: {pdf_type}")
    print(f"   📄 总页数: {result['pages']}")
    print(f"   📝 原始文本行数: {result['raw_lines']}")
    print(f"   📝 处理后段落数: {result['paragraphs']}")
    print(f"   📷 图片数: {result['images']}")
    print(f"   📊 表格数: {result['tables']}")
    print(f"   📁 输出目录: {output_subdir}")
    
    return result

# 用法示例
if __name__ == "__main__":
    # 替换为你的PDF文件路径
    pdf_path = r"/workspace/CN201923601U_de.pdf"
    
    # 处理PDF
    result = smart_extract_pdf(pdf_path)    