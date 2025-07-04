import os
import json
import re
import glob
from vector_utils import load_texts_from_output, retrieve

def extract_page_numbers(question):
    """从问题中提取页码信息"""
    page_patterns = [
        r'第(\d+)页',
        r'第(\d+)页的',
        r'页码(\d+)',
        r'(\d+)页'
    ]
    
    pages = []
    for pattern in page_patterns:
        matches = re.findall(pattern, question)
        pages.extend([int(match) for match in matches])
    
    return list(set(pages))  # 去重

def find_figure_files(doc_output_dir, pages):
    """根据页码查找对应的图表文件"""
    figure_files = []
    
    if not pages:
        return figure_files
    
    for page_num in pages:
        # 查找包含指定页码的图表文件
        pattern = os.path.join(doc_output_dir, f"*page{page_num}*.png")
        files = glob.glob(pattern)
        figure_files.extend(files)
    
    return figure_files

def test_single_document(doc_name, question, top_k=3):
    """
    测试单个文档的检索功能
    
    Args:
        doc_name: 文档名称（不包含.pdf后缀）
        question: 测试问题
        top_k: 返回的相关文本数量
    """
    print(f"📄 测试文档: {doc_name}")
    print(f"🔍 测试问题: {question}")
    print("-" * 80)
    
    # 构建文档输出目录路径
    doc_output_dir = os.path.join("output", doc_name)
    
    # 检查目录是否存在
    if not os.path.exists(doc_output_dir):
        print(f"❌ 错误: 未找到文档目录 {doc_output_dir}")
        return
    
    # 加载文本
    print("📚 正在加载文本...")
    texts = load_texts_from_output(doc_output_dir)
    
    if not texts:
        print("❌ 错误: 未找到文本内容")
        return
    
    print(f"✅ 成功加载 {len(texts)} 行文本")
    
    # 检索相关文本
    print(f"\n🔍 正在检索相关文本 (top_k={top_k})...")
    retrieved_texts = retrieve(texts, question, top_k=top_k)
    
    print(f"✅ 检索完成，找到 {len(retrieved_texts)} 条相关文本")
    
    # 显示检索结果
    print(f"\n📋 检索结果:")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"\n{i}. {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # 检查是否包含页码信息
    pages = extract_page_numbers(question)
    if pages:
        print(f"\n📄 检测到页码: {pages}")
        figure_files = find_figure_files(doc_output_dir, pages)
        if figure_files:
            print(f"📊 找到相关图片:")
            for fig in figure_files:
                print(f"   - {os.path.basename(fig)}")
        else:
            print(f"⚠️ 未找到页码 {pages} 对应的图片文件")
    else:
        print(f"\n📄 问题中未包含页码信息")
    
    return {
        'retrieved_texts': retrieved_texts,
        'figure_files': figure_files if pages else [],
        'pages': pages
    }

def test_multiple_questions():
    """测试多个问题的检索效果"""
    
    # 测试用例
    test_cases = [
        {
            "doc_name": "CN212149980U",  # 请替换为实际的文档名
            "questions": [
                "在文件中第6页的图片中，部件21相对于部件11的位置关系是什么？"
            ]
        }
    ]
    
    print("🚀 开始批量检索测试")
    print("=" * 100)
    
    for case in test_cases:
        doc_name = case["doc_name"]
        questions = case["questions"]
        
        print(f"\n📁 测试文档组: {doc_name}")
        print(f"📝 测试问题数: {len(questions)}")
        print("-" * 80)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔹 测试 {i}/{len(questions)}")
            result = test_single_document(doc_name, question, top_k=3)
            
            if result:
                print(f"✅ 检索成功")
                if result['pages']:
                    print(f"📄 包含页码: {result['pages']}")
                if result['figure_files']:
                    print(f"📊 相关图片: {len(result['figure_files'])} 张")
            else:
                print(f"❌ 检索失败")
            
            print("-" * 50)

def interactive_test():
    """交互式测试模式"""
    print("🎯 交互式检索测试")
    print("输入 'quit' 退出")
    print("-" * 50)
    
    # 获取可用的文档列表
    output_dir = "output"
    if os.path.exists(output_dir):
        docs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d))]
        if docs:
            print(f"📁 可用文档: {', '.join(docs)}")
        else:
            print("⚠️ 未找到已处理的文档")
            return
    else:
        print("❌ 输出目录不存在")
        return
    
    while True:
        print("\n" + "="*50)
        
        # 输入文档名
        doc_name = input("📄 请输入文档名称: ").strip()
        if doc_name.lower() == 'quit':
            break
        
        if doc_name not in docs:
            print(f"❌ 文档 '{doc_name}' 不存在，可用文档: {', '.join(docs)}")
            continue
        
        # 输入问题
        question = input("🔍 请输入问题: ").strip()
        if question.lower() == 'quit':
            break
        
        if not question:
            print("⚠️ 问题不能为空")
            continue
        
        # 执行检索
        try:
            result = test_single_document(doc_name, question)
            print(f"\n✅ 检索完成!")
        except Exception as e:
            print(f"❌ 检索出错: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检索功能测试脚本')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'], 
                       default='interactive', help='测试模式')
    parser.add_argument('--doc', help='文档名称（single模式使用）')
    parser.add_argument('--question', help='测试问题（single模式使用）')
    parser.add_argument('--top-k', type=int, default=3, help='返回结果数量')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.doc or not args.question:
            print("❌ single模式需要指定 --doc 和 --question 参数")
            return
        test_single_document(args.doc, args.question, args.top_k)
    
    elif args.mode == 'batch':
        test_multiple_questions()
    
    elif args.mode == 'interactive':
        interactive_test()

if __name__ == "__main__":
    main()
