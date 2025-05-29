import os
import cv2
import pdfplumber
import numpy as np

def extract_first_page_figure(pdf_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        # 读取第一页
        page = pdf.pages[0]
        pil_img = page.to_image(resolution=300).original.convert("RGB")
        
        # 将PIL图像转换为OpenCV的BGR格式
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]

        # 假设附图位于页面的右下角，定义一个区域（如右下角的一个矩形区域）
        # 我们取页面的右下角区域 (假设该附图占页面右下角的1/4区域)
        # 你可以根据实际情况调整这个区域的尺寸
        x_start = int(width * 0.5)  # 右下角区域的起始 x 坐标
        y_start = int(height * 0.5)  # 右下角区域的起始 y 坐标
        x_end = width  # 右下角区域的结束 x 坐标
        y_end = height  # 右下角区域的结束 y 坐标

        # 裁剪右下角区域
        cropped = img_cv[y_start:y_end, x_start:x_end]

        # 如果需要进一步处理（例如，通过阈值处理提取附图），可以加入以下代码：
        # 转为灰度图像进行进一步处理（例如二值化或边缘检测）
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # 阈值处理，调整阈值可能需要

        # 保存裁剪的附图
        out_path = os.path.abspath(os.path.join(output_dir, "page1.png"))
        cv2.imwrite(out_path, cropped)
        print(f"✅ 提取的附图保存至: {out_path}")

if __name__ == "__main__":
    extract_first_page_figure(
        pdf_path=r"/home/zhanggu/Project/tianchi/split_pdfs/output/front.pdf",
        output_dir="front"
    )
