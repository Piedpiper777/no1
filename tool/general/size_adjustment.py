import fitz
import os
from collections import defaultdict

def analyze_header_footer_height(pdf_path, sample_pages=10):
    """分析PDF文件，确定页眉和页脚区域的高度"""
    doc = fitz.open(pdf_path)

    # 样本页数
    total_pages = min(sample_pages, doc.page_count)

    # 用于存储每页不同Y坐标处的文本出现频率
    header_text_positions = defaultdict(int)
    footer_text_positions = defaultdict(int)

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_height = page.rect.height

        # 获取页面上的所有文本块
        text_blocks = page.get_text("blocks")

        for b in text_blocks:
            bbox = b[:4]  # 块的边界框：x0, y0, x1, y1
            text = b[4]  # 块中的文本

            # 计算文本块顶部和底部的Y坐标百分比
            y0_percent = bbox[1] / page_height
            y1_percent = bbox[3] / page_height

            # 页眉区域（页面顶部）
            if y0_percent < 0.2:  # 只考虑页面顶部20%的区域
                header_text_positions[(y0_percent, y1_percent)] += 1

            # 页脚区域（页面底部）
            if y1_percent > 0.8:  # 只考虑页面底部20%的区域
                footer_text_positions[(y0_percent, y1_percent)] += 1

    doc.close()

    # 确定页眉高度
    header_height = 0.0
    if header_text_positions:
        # 按出现频率排序
        sorted_header_positions = sorted(header_text_positions.items(),
                                         key=lambda x: x[1], reverse=True)
        # 取最常出现的位置作为页眉区域
        header_y0, header_y1 = sorted_header_positions[0][0]
        header_height = header_y1 - header_y0

    # 确定页脚高度
    footer_height = 0.0
    if footer_text_positions:
        sorted_footer_positions = sorted(footer_text_positions.items(),
                                         key=lambda x: x[1], reverse=True)
        footer_y0, footer_y1 = sorted_footer_positions[0][0]
        footer_height = footer_y1 - footer_y0

    return header_height, footer_height


def crop_pdf(pdf_path, output_path, header_height_percent, footer_height_percent):
    """裁剪PDF文件的页眉和页脚"""
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_height = page.rect.height
        header_height = header_height_percent * page_height
        footer_height = footer_height_percent * page_height

        new_rect = fitz.Rect(
            page.rect.x0,
            page.rect.y0 + header_height,
            page.rect.x1,
            page.rect.y1 - footer_height
        )
        page.set_cropbox(new_rect)
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    new_doc.save(output_path)
    doc.close()
    new_doc.close()


if __name__ == "__main__":
    pdf_path = r"/workspace/split_pdfs/CN111964678B/claims/claims.pdf"
    if not os.path.exists(pdf_path):
        print(f"错误: 文件 {pdf_path} 不存在")
        exit(1)

    suggested_header_height, suggested_footer_height = analyze_header_footer_height(pdf_path)
    print(f"建议的页眉裁剪高度占比: {suggested_header_height:.2%} of page height")
    print(f"建议的页脚裁剪高度占比: {suggested_footer_height:.2%} of page height")

    header_height_percent = float(input("请输入页眉裁剪高度占页面高度的比例（例如0.1表示10%）: "))
    footer_height_percent = float(input("请输入页脚裁剪高度占页面高度的比例（例如0.1表示10%）: "))

    output_dir = os.path.dirname(pdf_path)
    output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + "_cropped.pdf"
    output_path = os.path.join(output_dir, output_filename)

    crop_pdf(pdf_path, output_path, header_height_percent, footer_height_percent)
    print(f"裁剪后的PDF已保存至: {output_path}")