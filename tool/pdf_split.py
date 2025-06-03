import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import argparse
from typing import List, Tuple, Dict
import re
<<<<<<< HEAD


class PatentPDFSplitter:
    def __init__(self, use_gpu=False, match_algorithm='v3', max_chinese_chars=10, use_continuity_rules=True):
        """
        初始化专利PDF分割器
        
        Args:
            use_gpu: 是否使用GPU加速PaddleOCR
            match_algorithm: 匹配算法版本 ('v2', 'v3')
            max_chinese_chars: 只考虑OCR结果的前k个汉字 (0表示不限制)
            use_continuity_rules: 是否使用章节连续机制
        """
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu)
        
        # 匹配算法版本
        self.match_algorithm = match_algorithm
        
        # 只考虑前k个汉字
        self.max_chinese_chars = max_chinese_chars
        
        # 章节连续机制
        self.use_continuity_rules = use_continuity_rules
        
        # 定义各部分的页眉关键词（按优先级排序）
        self.header_keywords = {
            'drawings': ['附', '图', '附图', '说明书附图'],
            'descriptions': ['说', '明', '说明书', '说明', '明书'],
            'claims': ['要', '求', '要求', '权利', '权利要求', '权利要求书'],
            'front': ['国', '家', '国家', '知识', '产权', '知识产权', '国家知识产权局']
        }
        
        # 定义匹配优先级（避免混淆，优先匹配更具体的）
        self.match_priority = ['drawings', 'claims', 'descriptions', 'front']
        
        # 页眉检测区域比例（页面顶部的比例）
        self.header_region_ratio = 0.15
        
    def extract_chinese_chars(self, text: str, max_chars: int = 0) -> str:
        """
        提取文本中的汉字，可选择只保留前k个汉字
        
        Args:
            text: 输入文本
            max_chars: 最大汉字数量，0表示不限制
            
        Returns:
            提取的汉字字符串
        """
        # 使用正则表达式匹配汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        if max_chars > 0:
            chinese_chars = chinese_chars[:max_chars]
        
        return ''.join(chinese_chars)
    
    def pdf_page_to_image(self, page) -> np.ndarray:
        """
        将PDF页面转换为图像
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            numpy数组格式的图像
        """
        # 设置较高的分辨率以提高OCR准确性
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    
    def extract_header_region(self, img: np.ndarray) -> np.ndarray:
        """
        提取页面顶部的页眉区域
        
        Args:
            img: 输入图像
            
        Returns:
            页眉区域图像
        """
        height, width = img.shape[:2]
        header_height = int(height * self.header_region_ratio)
        
        # 提取顶部区域作为页眉
        header_img = img[0:header_height, :]
        
        return header_img
    
    def recognize_text(self, img: np.ndarray) -> List[str]:
        """
        使用PaddleOCR识别图像中的文本
        
        Args:
            img: 输入图像
            
        Returns:
            识别出的文本列表
        """
        try:
            result = self.ocr.ocr(img, cls=True)
            
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        texts.append(text)
            
            return texts
        except Exception as e:
            print(f"OCR识别错误: {e}")
            return []
    
    def find_best_match(self, text: str, keywords: List[str]) -> Tuple[bool, int, str]:
        """
        在文本中查找最佳匹配的关键词
        
        Args:
            text: 待匹配的文本
            keywords: 关键词列表
            
        Returns:
            (是否匹配, 匹配位置, 匹配的关键词)
        """
        best_match = None
        best_pos = len(text)
        best_keyword = ""
        
        for keyword in keywords:
            pos = text.find(keyword)
            if pos != -1 and pos < best_pos:
                best_match = True
                best_pos = pos
                best_keyword = keyword
        
        return (best_match is not None, best_pos, best_keyword)
    
    def classify_page_type_v2(self, texts: List[str]) -> str:
        """
        改进版页面类型分类，使用位置匹配算法
        
        Args:
            texts: 识别出的文本列表
            
        Returns:
            页面类型 ('front', 'claims', 'descriptions', 'drawings', 'unknown')
        """
        # 将所有文本合并，提取汉字（如果设置了限制）
        combined_text = ''.join(texts).replace(' ', '').replace('　', '')
        
        if self.max_chinese_chars > 0:
            combined_text = self.extract_chinese_chars(combined_text, self.max_chinese_chars)
            print(f"  提取前{self.max_chinese_chars}个汉字: '{combined_text}'")
        else:
            print(f"  合并文本: '{combined_text}'")
        
        # 按优先级匹配，避免混淆
        match_results = {}
        
        for page_type in self.match_priority:
            keywords = self.header_keywords[page_type]
            is_match, pos, matched_keyword = self.find_best_match(combined_text, keywords)
            
            if is_match:
                match_results[page_type] = {
                    'position': pos,
                    'keyword': matched_keyword,
                    'score': len(matched_keyword)  # 长度作为匹配强度
                }
                print(f"    {page_type}: 匹配到 '{matched_keyword}' 位置 {pos}")
        
        if not match_results:
            print(f"    未匹配到任何关键词")
            return 'unknown'
        
        # 特殊处理：如果同时匹配到"说明书"相关和"附图"相关，优先选择"附图"
        if 'drawings' in match_results and 'descriptions' in match_results:
            drawings_pos = match_results['drawings']['position']
            desc_pos = match_results['descriptions']['position']
            
            # 如果"附图"相关词汇出现在"说明书"相关词汇之后，且距离较近，判断为附图
            if abs(drawings_pos - desc_pos) <= 10:  # 距离阈值
                print(f"    检测到'说明书附图'组合，分类为 drawings")
                return 'drawings'
        
        # 选择位置最靠前且匹配强度最高的
        best_match = min(match_results.items(), 
                        key=lambda x: (x[1]['position'], -x[1]['score']))
        
        result_type = best_match[0]
        print(f"    最终分类: {result_type} (关键词: '{best_match[1]['keyword']}')")
        
        return result_type

    def char_level_match(self, text: str, keywords: List[str]) -> Tuple[bool, float, str]:
        """
        字符级别的模糊匹配，处理OCR识别不完整的情况
        
        Args:
            text: 待匹配的文本
            keywords: 关键词列表
            
        Returns:
            (是否匹配, 匹配得分, 最佳匹配关键词)
        """
        best_score = 0
        best_keyword = ""
        
        for keyword in keywords:
            # 计算字符匹配度
            if len(keyword) == 1:
                # 单字符匹配
                if keyword in text:
                    score = 1.0
                else:
                    score = 0
            else:
                # 多字符匹配，计算包含的字符比例
                matched_chars = sum(1 for char in keyword if char in text)
                score = matched_chars / len(keyword)
                
                # 如果是完全匹配，给予额外加分
                if keyword in text:
                    score += 0.5
                    
                # 如果字符是连续出现的，给予额外加分
                for i in range(len(text) - len(keyword) + 1):
                    substring = text[i:i+len(keyword)]
                    if substring == keyword:
                        score += 1.0
                        break
                    # 检查部分连续匹配
                    partial_match = 0
                    for j, char in enumerate(keyword):
                        if j < len(substring) and substring[j] == char:
                            partial_match += 1
                        else:
                            break
                    if partial_match > len(keyword) * 0.6:  # 60%以上连续匹配
                        score += partial_match / len(keyword) * 0.3
            
            if score > best_score:
                best_score = score
                best_keyword = keyword
        
        # 设置匹配阈值
        threshold = 0.3 if len(best_keyword) > 1 else 1.0
        is_match = best_score >= threshold
        
        return (is_match, best_score, best_keyword)
    
    def classify_page_type_v3(self, texts: List[str]) -> str:
        """
        使用字符级匹配的页面分类方法
        
        Args:
            texts: 识别出的文本列表
            
        Returns:
            页面类型
        """
        # 将所有文本合并，去除空格和标点符号
        combined_text = ''.join(texts).replace(' ', '').replace('　', '').replace(',', '').replace('，', '')
        
        # 如果设置了汉字限制，只保留前k个汉字
        if self.max_chinese_chars > 0:
            combined_text = self.extract_chinese_chars(combined_text, self.max_chinese_chars)
            print(f"  提取前{self.max_chinese_chars}个汉字: '{combined_text}'")
        else:
            print(f"  合并文本: '{combined_text}'")
        
        # 对每个类型进行字符级匹配
        type_scores = {}
        
        for page_type in self.match_priority:
            keywords = self.header_keywords[page_type]
            is_match, score, matched_keyword = self.char_level_match(combined_text, keywords)
            
            if is_match:
                type_scores[page_type] = {
                    'score': score,
                    'keyword': matched_keyword
                }
                print(f"    {page_type}: 匹配 '{matched_keyword}' 得分 {score:.2f}")
        
        if not type_scores:
            print(f"    未匹配到任何关键词")
            return 'unknown'
        
        # 特殊处理逻辑：处理"说明书附图"和"说明书"的混淆
        if 'drawings' in type_scores and 'descriptions' in type_scores:
            drawings_score = type_scores['drawings']['score']
            desc_score = type_scores['descriptions']['score']
            
            # 检查是否包含"附图"的特征字符
            has_attachment_chars = any(char in combined_text for char in ['附', '图'])
            
            if has_attachment_chars and drawings_score >= desc_score * 0.7:
                print(f"    检测到附图特征，优先选择 drawings")
                return 'drawings'
        
        # 选择得分最高的类型
        best_type = max(type_scores.items(), key=lambda x: x[1]['score'])
        result_type = best_type[0]
        
        print(f"    最终分类: {result_type} (关键词: '{best_type[1]['keyword']}', 得分: {best_type[1]['score']:.2f})")
        
        return result_type
    
    def check_continuity_errors(self, page_types: List[str]) -> List[Dict]:
        errors = []
        total_pages = len(page_types)
        print("\n检查章节连续性错误...")

    # 只保留：孤立页面检测
        for i in range(1, total_pages - 1):
            current_type = page_types[i]
            prev_type = page_types[i - 1]
            next_type = page_types[i + 1]

            if current_type != prev_type and current_type != next_type and prev_type == next_type:
                errors.append({
                    'type': 'isolated_page',
                    'page': i + 1,
                    'current_type': current_type,
                    'surrounding_type': prev_type,
                    'confidence': 0.8,
                    'description': f'页面{i + 1}被{prev_type}包围，但分类为{current_type}'
                })

        return errors

    
    def apply_continuity_corrections(self, page_types: List[str], errors: List[Dict]) -> List[str]:
        """
        根据连续性检查结果应用纠正
        
        Args:
            page_types: 原始页面类型列表
            errors: 错误报告列表
            
        Returns:
            纠正后的页面类型列表
        """
        if not self.use_continuity_rules:
            return page_types
        
        print("\n应用章节连续性纠正...")
        
        result = page_types.copy()
        corrections_made = []
        
        # 按置信度排序，优先处理高置信度的错误
        sorted_errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)
        
        for error in sorted_errors:
            page_idx = error['page'] - 1
            current_type = result[page_idx]

            if error['type'] == 'isolated_page' and error['confidence'] >= 0.7:
                surrounding_type = error['surrounding_type']
                result[page_idx] = surrounding_type
                corrections_made.append({
                    'page': error['page'],
                    'from': current_type,
                    'to': surrounding_type,
                    'reason': '孤立页面纠正'
                })
                print(f"    纠正页面 {error['page']}: {current_type} -> {surrounding_type} (孤立页面纠正)")

        
        # 处理剩余的unknown页面
        for i in range(len(result)):
            if result[i] == 'unknown':
                # 查找最近的已知类型
                nearest_type = 'front'  # 默认值
                
                # 优先查找前面的类型
                for j in range(i-1, -1, -1):
                    if result[j] != 'unknown':
                        nearest_type = result[j]
                        break
                
                # 如果前面没有，查找后面的类型
                if nearest_type == 'front':
                    for j in range(i+1, len(result)):
                        if result[j] != 'unknown':
                            nearest_type = result[j]
                            break
                
                result[i] = nearest_type
                corrections_made.append({
                    'page': i + 1,
                    'from': 'unknown',
                    'to': nearest_type,
                    'reason': '未识别页面处理'
                })
                print(f"    处理未识别页面 {i+1}: unknown -> {nearest_type}")
        
        # 输出纠正统计
        if corrections_made:
            print(f"\n总共进行了 {len(corrections_made)} 次纠正:")
            correction_stats = {}
            for correction in corrections_made:
                reason = correction['reason']
                correction_stats[reason] = correction_stats.get(reason, 0) + 1
            
            for reason, count in correction_stats.items():
                print(f"  {reason}: {count} 次")
        else:
            print("\n没有进行任何纠正")
        return result
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, List[int]]:
        """
        分析PDF结构，确定各部分的页面范围
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            各部分对应的页面列表字典
        """
        doc = fitz.open(pdf_path)
        page_types = []
        
        print(f"开始分析PDF结构，总页数: {len(doc)}")
        if self.max_chinese_chars > 0:
            print(f"汉字限制: 只考虑前 {self.max_chinese_chars} 个汉字")
        if self.use_continuity_rules:
            print(f"章节连续性: 已启用")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
=======


class PatentPDFSplitter:
    def __init__(self, use_gpu=False, match_algorithm='v3'):
        """
        初始化专利PDF分割器
        
        Args:
            use_gpu: 是否使用GPU加速PaddleOCR
            match_algorithm: 匹配算法版本 ('v2', 'v3')
        """
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu)
        
        # 匹配算法版本
        self.match_algorithm = match_algorithm
        
        # 定义各部分的页眉关键词（按优先级排序）
        self.header_keywords = {
            'drawings': ['附', '图', '附图', '说明书附图'],
            'descriptions': ['说', '明', '说明书', '说明', '明书'],
            'claims': ['要', '求', '要求', '权利', '权利要求', '权利要求书'],
            'front': ['国', '家', '国家', '知识', '产权', '知识产权', '国家知识产权局']
        }
        
        # 定义匹配优先级（避免混淆，优先匹配更具体的）
        self.match_priority = ['drawings', 'claims', 'descriptions', 'front']
        
        # 页眉检测区域比例（页面顶部的比例）
        self.header_region_ratio = 0.15
        
    def pdf_page_to_image(self, page) -> np.ndarray:
        """
        将PDF页面转换为图像
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            numpy数组格式的图像
        """
        # 设置较高的分辨率以提高OCR准确性
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    
    def extract_header_region(self, img: np.ndarray) -> np.ndarray:
        """
        提取页面顶部的页眉区域
        
        Args:
            img: 输入图像
            
        Returns:
            页眉区域图像
        """
        height, width = img.shape[:2]
        header_height = int(height * self.header_region_ratio)
        
        # 提取顶部区域作为页眉
        header_img = img[0:header_height, :]
        
        return header_img
    
    def recognize_text(self, img: np.ndarray) -> List[str]:
        """
        使用PaddleOCR识别图像中的文本
        
        Args:
            img: 输入图像
            
        Returns:
            识别出的文本列表
        """
        try:
            result = self.ocr.ocr(img, cls=True)
            
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        texts.append(text)
            
            return texts
        except Exception as e:
            print(f"OCR识别错误: {e}")
            return []
    
    def find_best_match(self, text: str, keywords: List[str]) -> Tuple[bool, int, str]:
        """
        在文本中查找最佳匹配的关键词
        
        Args:
            text: 待匹配的文本
            keywords: 关键词列表
            
        Returns:
            (是否匹配, 匹配位置, 匹配的关键词)
        """
        best_match = None
        best_pos = len(text)
        best_keyword = ""
        
        for keyword in keywords:
            pos = text.find(keyword)
            if pos != -1 and pos < best_pos:
                best_match = True
                best_pos = pos
                best_keyword = keyword
        
        return (best_match is not None, best_pos, best_keyword)
    
    def classify_page_type_v2(self, texts: List[str]) -> str:
        """
        改进版页面类型分类，使用位置匹配算法
        
        Args:
            texts: 识别出的文本列表
            
        Returns:
            页面类型 ('front', 'claims', 'descriptions', 'drawings', 'unknown')
        """
        # 将所有文本合并，去除空格和标点符号
        combined_text = ''.join(texts).replace(' ', '').replace('　', '')
        
        print(f"  合并文本: '{combined_text}'")
        
        # 按优先级匹配，避免混淆
        match_results = {}
        
        for page_type in self.match_priority:
            keywords = self.header_keywords[page_type]
            is_match, pos, matched_keyword = self.find_best_match(combined_text, keywords)
            
            if is_match:
                match_results[page_type] = {
                    'position': pos,
                    'keyword': matched_keyword,
                    'score': len(matched_keyword)  # 长度作为匹配强度
                }
                print(f"    {page_type}: 匹配到 '{matched_keyword}' 位置 {pos}")
        
        if not match_results:
            print(f"    未匹配到任何关键词")
            return 'unknown'
        
        # 特殊处理：如果同时匹配到"说明书"相关和"附图"相关，优先选择"附图"
        if 'drawings' in match_results and 'descriptions' in match_results:
            drawings_pos = match_results['drawings']['position']
            desc_pos = match_results['descriptions']['position']
            
            # 如果"附图"相关词汇出现在"说明书"相关词汇之后，且距离较近，判断为附图
            if abs(drawings_pos - desc_pos) <= 10:  # 距离阈值
                print(f"    检测到'说明书附图'组合，分类为 drawings")
                return 'drawings'
        
        # 选择位置最靠前且匹配强度最高的
        best_match = min(match_results.items(), 
                        key=lambda x: (x[1]['position'], -x[1]['score']))
        
        result_type = best_match[0]
        print(f"    最终分类: {result_type} (关键词: '{best_match[1]['keyword']}')")
        
        return result_type

    def char_level_match(self, text: str, keywords: List[str]) -> Tuple[bool, float, str]:
        """
        字符级别的模糊匹配，处理OCR识别不完整的情况
        
        Args:
            text: 待匹配的文本
            keywords: 关键词列表
            
        Returns:
            (是否匹配, 匹配得分, 最佳匹配关键词)
        """
        best_score = 0
        best_keyword = ""
        
        for keyword in keywords:
            # 计算字符匹配度
            if len(keyword) == 1:
                # 单字符匹配
                if keyword in text:
                    score = 1.0
                else:
                    score = 0
            else:
                # 多字符匹配，计算包含的字符比例
                matched_chars = sum(1 for char in keyword if char in text)
                score = matched_chars / len(keyword)
                
                # 如果是完全匹配，给予额外加分
                if keyword in text:
                    score += 0.5
                    
                # 如果字符是连续出现的，给予额外加分
                for i in range(len(text) - len(keyword) + 1):
                    substring = text[i:i+len(keyword)]
                    if substring == keyword:
                        score += 1.0
                        break
                    # 检查部分连续匹配
                    partial_match = 0
                    for j, char in enumerate(keyword):
                        if j < len(substring) and substring[j] == char:
                            partial_match += 1
                        else:
                            break
                    if partial_match > len(keyword) * 0.6:  # 60%以上连续匹配
                        score += partial_match / len(keyword) * 0.3
            
            if score > best_score:
                best_score = score
                best_keyword = keyword
        
        # 设置匹配阈值
        threshold = 0.3 if len(best_keyword) > 1 else 1.0
        is_match = best_score >= threshold
        
        return (is_match, best_score, best_keyword)
    
    def classify_page_type_v3(self, texts: List[str]) -> str:
        """
        使用字符级匹配的页面分类方法
        
        Args:
            texts: 识别出的文本列表
            
        Returns:
            页面类型
        """
        # 将所有文本合并，去除空格和标点符号
        combined_text = ''.join(texts).replace(' ', '').replace('　', '').replace(',', '').replace('，', '')
        
        print(f"  合并文本: '{combined_text}'")
        
        # 对每个类型进行字符级匹配
        type_scores = {}
        
        for page_type in self.match_priority:
            keywords = self.header_keywords[page_type]
            is_match, score, matched_keyword = self.char_level_match(combined_text, keywords)
            
            if is_match:
                type_scores[page_type] = {
                    'score': score,
                    'keyword': matched_keyword
                }
                print(f"    {page_type}: 匹配 '{matched_keyword}' 得分 {score:.2f}")
        
        if not type_scores:
            print(f"    未匹配到任何关键词")
            return 'unknown'
        
        # 特殊处理逻辑：处理"说明书附图"和"说明书"的混淆
        if 'drawings' in type_scores and 'descriptions' in type_scores:
            drawings_score = type_scores['drawings']['score']
            desc_score = type_scores['descriptions']['score']
            
            # 检查是否包含"附图"的特征字符
            has_attachment_chars = any(char in combined_text for char in ['附', '图'])
            
            if has_attachment_chars and drawings_score >= desc_score * 0.7:
                print(f"    检测到附图特征，优先选择 drawings")
                return 'drawings'
        
        # 选择得分最高的类型
        best_type = max(type_scores.items(), key=lambda x: x[1]['score'])
        result_type = best_type[0]
        
        print(f"    最终分类: {result_type} (关键词: '{best_type[1]['keyword']}', 得分: {best_type[1]['score']:.2f})")
        
        return result_type
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, List[int]]:
        """
        分析PDF结构，确定各部分的页面范围
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            各部分对应的页面列表字典
        """
        doc = fitz.open(pdf_path)
        page_types = []
        
        print(f"开始分析PDF结构，总页数: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
>>>>>>> 38bed604146978c80a320ab4dd64d6702ed354af
            # 转换页面为图像
            img = self.pdf_page_to_image(page)
            
            # 提取页眉区域
            header_img = self.extract_header_region(img)
            
            # 识别页眉文本
            texts = self.recognize_text(header_img)
            
            # 分类页面类型 - 根据选择的算法版本
            if self.match_algorithm == 'v3':
                page_type = self.classify_page_type_v3(texts)
            else:
                page_type = self.classify_page_type_v2(texts)
            page_types.append(page_type)
            
            print(f"页面 {page_num + 1}: 识别文本 {texts} -> 类型: {page_type}")
        
        doc.close()
        
<<<<<<< HEAD
        # 执行后处理检查和纠正
        if self.use_continuity_rules:
            # 检查连续性错误
            errors = self.check_continuity_errors(page_types)
            
            if errors:
                print(f"\n发现 {len(errors)} 个潜在错误:")
                for error in errors:
                    if error['type'] == 'isolated_page':
                        print(f"  {error['description']} (置信度: {error['confidence']:.1f})")
                
                # 应用纠正
                page_types = self.apply_continuity_corrections(page_types, errors)
            else:
                print("\n未发现连续性错误")
        
=======
>>>>>>> 38bed604146978c80a320ab4dd64d6702ed354af
        # 组织各部分的页面范围
        sections = {'front': [], 'claims': [], 'descriptions': [], 'drawings': []}
        
        for i, page_type in enumerate(page_types):
<<<<<<< HEAD
            sections[page_type].append(i)
        
        # 显示最终结果
        print("\n最终页面分类结果:")
        for i, page_type in enumerate(page_types):
            print(f"页面 {i + 1}: {page_type}")
=======
            if page_type != 'unknown':
                sections[page_type].append(i)
        
        # 处理未识别的页面，根据上下文推断
        for i, page_type in enumerate(page_types):
            if page_type == 'unknown':
                # 查找前后已识别的页面类型
                prev_type = None
                next_type = None
                
                for j in range(i-1, -1, -1):
                    if page_types[j] != 'unknown':
                        prev_type = page_types[j]
                        break
                
                for j in range(i+1, len(page_types)):
                    if page_types[j] != 'unknown':
                        next_type = page_types[j]
                        break
                
                # 根据上下文推断类型
                if prev_type and not next_type:
                    # 如果只有前面的类型，继续使用前面的类型
                    inferred_type = prev_type
                elif prev_type == next_type:
                    # 如果前后类型相同，使用该类型
                    inferred_type = prev_type
                elif prev_type:
                    # 如果有前面的类型，使用前面的类型
                    inferred_type = prev_type
                else:
                    # 默认归类为首页
                    inferred_type = 'front'
                
                sections[inferred_type].append(i)
                print(f"页面 {i + 1}: 未识别类型，推断为 {inferred_type}")
>>>>>>> 38bed604146978c80a320ab4dd64d6702ed354af
        
        return sections
    
    def split_pdf(self, input_path: str, output_dir: str):
        """
        分割PDF文件
        
        Args:
            input_path: 输入PDF文件路径
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分析PDF结构
        sections = self.analyze_pdf_structure(input_path)
        
        # 打开原始PDF
        doc = fitz.open(input_path)
        
        # 生成各部分PDF
        section_names = {
            'front': 'front.pdf',
            'claims': 'claims.pdf', 
            'descriptions': 'descriptions.pdf',
            'drawings': 'drawings.pdf'
        }
        
        for section_type, pages in sections.items():
            if pages:  # 如果该部分有页面
                # 创建新的PDF文档
                new_doc = fitz.open()
                
                # 按页面顺序排序
                pages.sort()
                
                # 复制页面到新文档
                for page_num in pages:
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # 保存分割后的PDF
                output_path = os.path.join(output_dir, section_names[section_type])
                new_doc.save(output_path)
                new_doc.close()
                
                print(f"已生成 {section_type} 部分: {output_path} (页面: {[p+1 for p in pages]})")
            else:
                print(f"未找到 {section_type} 部分的页面")
        
        doc.close()
        print("PDF分割完成！")


def main():
    parser = argparse.ArgumentParser(description='专利PDF文件分割工具')
    parser.add_argument('input_pdf', help='输入的专利PDF文件路径')
    parser.add_argument('-o', '--output', default='./output', help='输出目录 (默认: ./output)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速PaddleOCR')
    parser.add_argument('--match-algorithm', choices=['v2', 'v3'], default='v3',
                       help='选择匹配算法: v2=位置匹配, v3=字符级匹配 (默认: v3)')
    parser.add_argument('--header-ratio', type=float, default=0.15, 
                       help='页眉区域占页面的比例 (默认: 0.15)')
<<<<<<< HEAD
    parser.add_argument('--max-chinese-chars', type=int, default=10,
                       help='只考虑OCR结果的前k个汉字，0表示不限制 (默认: 10)')
    parser.add_argument('--no-continuity', action='store_true',
                       help='禁用章节连续性检查和纠正')
=======
>>>>>>> 38bed604146978c80a320ab4dd64d6702ed354af
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_pdf):
        print(f"错误: 输入文件 {args.input_pdf} 不存在")
        return
    
    # 创建分割器
<<<<<<< HEAD
    splitter = PatentPDFSplitter(
        use_gpu=args.gpu, 
        match_algorithm=args.match_algorithm,
        max_chinese_chars=args.max_chinese_chars,
        use_continuity_rules=not args.no_continuity
    )
=======
    splitter = PatentPDFSplitter(use_gpu=args.gpu, match_algorithm=args.match_algorithm)
>>>>>>> 38bed604146978c80a320ab4dd64d6702ed354af
    
    # 设置页眉区域比例
    if args.header_ratio:
        splitter.header_region_ratio = args.header_ratio
    
    # 执行分割
    try:
        splitter.split_pdf(args.input_pdf, args.output)
    except Exception as e:
        print(f"分割过程中出现错误: {e}")


if __name__ == "__main__":
    main()