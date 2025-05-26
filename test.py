import os
import re
import cv2
import numpy as np
import pdfplumber
from PIL import Image
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
    """处理文本型PDF - 使用你的原始逻辑 + PPStructure增强"""
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

def extract_image_pdf(pdf_path, output_dir):
    """处理图片型PDF - 纯OCR + 结构化提取"""
    print("🖼️ 使用图片型PDF处理模式...")
    
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # 初始化OCR和结构分析
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    structure_engine = PPStructure(recovery=True, lang='ch', show_log=False)
    
    text_output = []
    img_counter = 0
    table_counter = 0
    current_id = None
    para_buffer = ""
    
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        print(f"📖 处理第 {page_num + 1} 页...")
        page = pdf_document[page_num]
        
        # 转换为高分辨率图片
        mat = fitz.Matrix(3.0, 3.0)  # 图片型PDF需要更高分辨率
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        try:
            # 使用PPStructure进行版面分析
            result = structure_engine(img)
            result.sort(key=lambda x: x['bbox'][1])  # 按y坐标排序
            
            for item in result:
                bbox = item['bbox']
                item_type = item['type']
                
                if item_type == 'text':
                    # 处理文本区域
                    text_content = item.get('res', [])
                    if isinstance(text_content, list):
                        page_text_lines = []
                        for text_item in text_content:
                            if isinstance(text_item, dict) and 'text' in text_item:
                                page_text_lines.append(text_item['text'])
                        
                        full_text = '\n'.join(page_text_lines)
                        
                        # 应用原有的段落识别逻辑
                        lines = full_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            match = re.match(r'\[(\d{4})\]', line)
                            if match:
                                if current_id is not None and para_buffer.strip():
                                    cleaned_para = fix_chinese_soft_breaks(para_buffer.strip())
                                    text_output.append(cleaned_para)
                                
                                current_id = match.group(1)
                                para_buffer = line[len(match.group(0)):].strip()
                            else:
                                para_buffer += ' ' + line.strip()
                
                elif item_type == 'figure':
                    # 处理图片
                    img_counter += 1
                    x0, y0, x1, y1 = [int(coord) for coord in bbox]
                    cropped_img = img[y0:y1, x0:x1]
                    
                    img_name = f"{current_id}_{img_counter}.png" if current_id else f"page{page_num+1}_img{img_counter}.png"
                    img_path = os.path.join(img_dir, img_name)
                    cv2.imwrite(img_path, cropped_img)
                    
                    if current_id:
                        para_buffer += f"\n[IMG_{img_counter}]"
                    print(f"📷 提取图片: {img_name}")
                
                elif item_type == 'table':
                    # 处理表格
                    table_counter += 1
                    x0, y0, x1, y1 = [int(coord) for coord in bbox]
                    cropped_table = img[y0:y1, x0:x1]
                    
                    table_name = f"{current_id}_table{table_counter}.png" if current_id else f"page{page_num+1}_table{table_counter}.png"
                    table_path = os.path.join(img_dir, table_name)
                    cv2.imwrite(table_path, cropped_table)
                    
                    if current_id:
                        para_buffer += f"\n[TABLE_{table_counter}]"
                    print(f"📊 提取表格: {table_name}")
        
        except Exception as e:
            print(f"⚠️ 第{page_num+1}页处理失败: {str(e)}")
            # 降级处理：使用纯OCR
            try:
                ocr_result = ocr_engine.ocr(img, cls=True)
                if ocr_result and ocr_result[0]:
                    page_text = []
                    for line in ocr_result[0]:
                        text = line[1][0]
                        page_text.append(text)
                    
                    full_text = '\n'.join(page_text)
                    lines = full_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        match = re.match(r'\[(\d{4})\]', line)
                        if match:
                            if current_id is not None and para_buffer.strip():
                                cleaned_para = fix_chinese_soft_breaks(para_buffer.strip())
                                text_output.append(cleaned_para)
                            
                            current_id = match.group(1)
                            para_buffer = line[len(match.group(0)):].strip()
                        else:
                            para_buffer += ' ' + line.strip()
                            
            except Exception as e2:
                print(f"❌ 纯OCR也失败了: {str(e2)}")
    
    # 保存最后一个段落
    if current_id and para_buffer.strip():
        cleaned_para = fix_chinese_soft_breaks(para_buffer.strip())
        text_output.append(cleaned_para)
    
    # 写入文本文件
    with open(os.path.join(output_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_output))
    
    pdf_document.close()
    return len(text_output), img_counter, table_counter

def smart_extract_pdf(pdf_path, output_dir):
    """智能PDF提取 - 自动判断类型并选择合适的处理方法"""
    
    print(f"🚀 开始智能处理PDF: {os.path.basename(pdf_path)}")
    
    # 第一步：检测PDF类型
    pdf_type = detect_pdf_type(pdf_path)
    
    # 第二步：选择对应的处理方法
    if pdf_type == 'text':
        paragraphs, images, tables = extract_text_pdf(pdf_path, output_dir)
    elif pdf_type == 'image':
        paragraphs, images, tables = extract_image_pdf(pdf_path, output_dir)
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
    pdf_path = r"/workspace/CN201923601U_de.pdf"
    output_dir = "output_smart1"
    
    try:
        result = smart_extract_pdf(pdf_path, output_dir)
        
        print(f"\n🎉 全部完成！输出文件：")
        print(f"   📄 text.txt - 结构化文本 ({result['paragraphs']} 段落)")
        print(f"   📁 images/ - {result['images']} 个图片 + {result['tables']} 个表格")
        print(f"   🔍 PDF类型: {result['pdf_type']}")
        
    except ImportError as e:
        print(f"❌ 依赖库缺失: {str(e)}")
        print("💡 请安装: pip install paddlepaddle paddleocr PyMuPDF opencv-python pdfplumber")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()