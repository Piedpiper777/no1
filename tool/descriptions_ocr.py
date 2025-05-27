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
        if debug:
            print(f"处理合并行: {line[:60]}...")

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
    """处理文本型PDF - 使用原始逻辑 + PPStructure增强"""
    print("📄 使用文本型PDF处理模式...")
    
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    text_output = []
    img_counter = 0
    table_counter = 0
    para_buffer = ""
    current_id = None
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            width, height = page.width, page.height
            crop = page.within_bbox((0, height * 0.07, width, height * 0.9))
            text = crop.extract_text()
            
            if not text:
                continue
            
            # 原有的文本处理逻辑
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'\[(\d{4})\]', line)
                if match:
                    if current_id is not None:
                        cleaned_para = fix_chinese_soft_breaks(para_buffer.strip())
                        text_output.append(cleaned_para)
                    current_id = match.group(1)
                    para_buffer = line[len(match.group(0)):].strip()
                else:
                    para_buffer += ' ' + line.strip()
            
            # PPStructure增强图片表格提取
            try:
                img_counter, table_counter, para_buffer = extract_images_tables_with_ppstructure(
                    pdf_path, page_num, current_id, img_counter, table_counter, img_dir, para_buffer
                )
            except Exception as e:
                print(f"⚠️ 页面{page_num+1}的PPStructure处理失败: {str(e)}")
    
    if current_id and para_buffer.strip():
        cleaned_para = fix_chinese_soft_breaks(para_buffer.strip())
        text_output.append(cleaned_para)
    
    with open(os.path.join(output_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_output))
    
    return len(text_output), img_counter, table_counter

def extract_image_pdf(pdf_path, output_dir, debug=False):
    """
    增强版图片型PDF处理器 - 集成自script2
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        debug: 是否输出调试信息和中间结果
    """
    
    print(f"🖼️ 使用增强版图片型PDF处理模式: {os.path.basename(pdf_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
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
        
        if debug:
            debug_img_path = os.path.join(debug_dir, f"page_{page_num+1}_original.png")
            cv2.imwrite(debug_img_path, original_img)
            print(f"🔍 原始图片已保存: {debug_img_path}")
        
        # 图片预处理
        processed_img = preprocess_image(original_img)
        
        if debug:
            debug_processed_path = os.path.join(debug_dir, f"page_{page_num+1}_processed.png")
            cv2.imwrite(debug_processed_path, processed_img)
            print(f"🔍 预处理图片已保存: {debug_processed_path}")
        
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
                    
                    if debug:
                        print(f"  发现{item_type}: {bbox}")
                    
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
                            
                            # 在文本中插入图片标记
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
                            
                            # 在文本中插入表格标记
                            page_text_lines.append(f"[TABLE_{table_counter}]")
                            print(f"  📊 提取表格: {table_name}")
        
        except Exception as e:
            print(f"⚠️ PPStructure分析失败: {str(e)}")
        
        # 方法2: 如果PPStructure没有成功提取文本，使用纯OCR
        if not structure_success:
            print("🔤 使用纯OCR模式...")
            try:
                # 使用预处理后的图片进行OCR
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
                            if debug:
                                print(f"  OCR: {text} (置信度: {confidence:.2f})")
                else:
                    print("  ⚠️ OCR返回空结果")
            
            except Exception as e:
                print(f"❌ OCR处理失败: {str(e)}")
        
        # 显示页面提取的文本预览
        if page_text_lines:
            page_text_preview = ' '.join(page_text_lines)
            if debug:
                print(f"  📝 页面文本预览: {page_text_preview[:100]}...")
            # 将页面文本添加到总文本中
            all_text_lines.extend(page_text_lines)
        else:
            print("  ⚠️ 本页未提取到文本")
    
    pdf_document.close()
    
    # 统一处理所有文本，应用段落识别逻辑
    print(f"\n🔄 处理提取的文本，共 {len(all_text_lines)} 行")
    
    if debug:
        print("📝 所有提取的文本行预览:")
        for i, line in enumerate(all_text_lines[:10]):  # 只显示前10行
            print(f"  {i+1}: {line}")
        if len(all_text_lines) > 10:
            print(f"  ... 还有 {len(all_text_lines)-10} 行")
    
    # 处理文本并识别段落
    text_output = process_text_with_paragraphs(all_text_lines, debug=debug)
    
    # 写入文本文件
    text_file = os.path.join(output_dir, "text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_output))  # 段落间用双换行分隔
    
    # 同时保存原始提取的文本（调试用）
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        raw_text_file = os.path.join(debug_dir, "raw_text.txt")
        with open(raw_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text_lines))
        print(f"🔍 原始文本已保存: {raw_text_file}")
    
    return len(text_output), img_counter, table_counter

def smart_extract_pdf(pdf_path, output_dir, debug=False):
    """智能PDF提取 - 自动判断类型并选择合适的处理方法"""
    
    print(f"🚀 开始智能处理PDF: {os.path.basename(pdf_path)}")
    
    # 第一步：检测PDF类型
    pdf_type = detect_pdf_type(pdf_path)
    
    # 第二步：选择对应的处理方法
    if pdf_type == 'text':
        paragraphs, images, tables = extract_text_pdf(pdf_path, output_dir)
    elif pdf_type == 'image':
        paragraphs, images, tables = extract_image_pdf(pdf_path, output_dir, debug=debug)
    else:  # mixed
        print("📄🖼️ 混合型PDF，使用文本模式处理（主要逻辑）+ OCR补充")
        # 混合型暂时使用文本模式，后续可以优化为逐页判断
        paragraphs, images, tables = extract_text_pdf(pdf_path, output_dir)
    
    print(f"\n✅ 处理完成！")
    print(f"   📊 PDF类型: {pdf_type}")
    print(f"   📄 段落数: {paragraphs}")
    print(f"   📷 图片数: {images}")
    print(f"   📋 表格数: {tables}")
    print(f"   📁 输出目录: {output_dir}")
    
    return {
        'pdf_type': pdf_type,
        'paragraphs': paragraphs,
        'images': images,
        'tables': tables
    }

# 用法示例
if __name__ == "__main__":
    pdf_path = r"/workspace/project/split_pdfs/CN111964678B/description/description.pdf"
    output_dir = "output_smart_integrated"
    
    try:
        # 开启调试模式以获得更详细的输出和中间文件
        result = smart_extract_pdf(pdf_path, output_dir, debug=True)
        
        print(f"\n🎉 全部完成！输出文件：")
        print(f"   📄 text.txt - 结构化文本 ({result['paragraphs']} 段落)")
        print(f"   📁 images/ - {result['images']} 个图片 + {result['tables']} 个表格")
        print(f"   🔍 PDF类型: {result['pdf_type']}")
        
        # 如果是调试模式，还会生成debug文件夹
        debug_dir = os.path.join(output_dir, "debug")
        if os.path.exists(debug_dir):
            print(f"   🔍 debug/ - 调试文件（原始图片、处理后图片、原始文本等）")
        
    except ImportError as e:
        print(f"❌ 依赖库缺失: {str(e)}")
        print("💡 请安装: pip install paddlepaddle paddleocr PyMuPDF opencv-python pdfplumber pillow")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()