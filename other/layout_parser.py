import os
import cv2
import pdf2image
import layoutparser as lp
from PIL import Image
import numpy as np

def pdf_to_images(pdf_path, dpi=300):
    """将PDF转换为图像列表"""
    return pdf2image.convert_from_path(pdf_path, dpi=dpi)

def analyze_layout(image):
    """分析图像布局，识别文字和图形区域"""
    # 使用PubLayNet预训练模型
    model = lp.Detectron2LayoutModel(
        config_path=r'/workspace/model/PubLayNet-faster_rcnn_R_50_FPN_3x/config.yml',
        model_path=r'/workspace/model/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

    # 转换为OpenCV格式
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 检测布局
    layout = model.detect(image_cv)
    
    # 筛选文字和图形区域
    text_regions = [b for b in layout if b.type in ['Text', 'Title', 'List']]
    figure_regions = [b for b in layout if b.type == 'Figure']
    
    return image_cv, text_regions, figure_regions

def crop_regions(image, regions, output_dir, prefix, extension='png'):
    """根据区域裁剪图像并保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, region in enumerate(regions):
        x_1, y_1, x_2, y_2 = region.coordinates
        cropped = image[int(y_1):int(y_2), int(x_1):int(x_2)]
        
        output_path = os.path.join(output_dir, f"{prefix}_{i}.{extension}")
        cv2.imwrite(output_path, cropped)
        print(f"已保存: {output_path}")

def process_patent_pdf(pdf_path, output_dir='output'):
    """处理专利PDF，分割文字和图形"""
    # 创建输出目录
    text_output = os.path.join(output_dir, 'text_regions')
    figure_output = os.path.join(output_dir, 'figure_regions')
    
    # 转换PDF为图像
    images = pdf_to_images(pdf_path)
    
    for i, image in enumerate(images):
        print(f"正在处理第 {i+1}/{len(images)} 页...")
        
        # 分析布局
        image_cv, text_regions, figure_regions = analyze_layout(image)
        
        # 裁剪并保存文字区域
        crop_regions(image_cv, text_regions, text_output, f"page_{i+1}_text")
        
        # 裁剪并保存图形区域
        crop_regions(image_cv, figure_regions, figure_output, f"page_{i+1}_figure")
    
    print(f"处理完成！文字区域保存在: {text_output}")
    print(f"图形区域保存在: {figure_output}")

if __name__ == "__main__":
    # 直接在代码中定义输入输出路径
    PDF_PATH = r"/workspace/pdf_files/CN218108941U.pdf"  # 输入PDF路径
    OUTPUT_DIR = r"/workspace/output/0"      # 输出目录路径
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    process_patent_pdf(PDF_PATH, OUTPUT_DIR)